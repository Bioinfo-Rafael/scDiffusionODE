import copy
import functools
import os
import glob

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        model_name,
        save_dir,
        ode_reg_lambda: float = 0.0, ode_reg_norm: str = 'l1',
        save_loss_details: bool = True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        """
        追加
        →soft constraint用

        ode_reg_lambda : float
            ODE正則化項のハイパラ
        ode_reg_norm : str
            'l1' or 'l2'
        save_loss_details : bool
            損失の詳細記録を保存するかどうか
        """
        self.ode_reg_lambda = ode_reg_lambda
        self.ode_reg_norm = ode_reg_norm
        
        # 損失記録用
        self.save_loss_details = save_loss_details



        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and dist.is_available() and dist.is_initialized():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist_util.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.timestamp = model_name #time.strftime("%m-%d-%H:%M",time.gmtime())
        
        # 既存のsave_dirが存在する場合、番号を付けて上書きを防ぐ
        original_save_dir = save_dir
        if os.path.exists(os.path.join(save_dir, model_name)):
            # 既存の番号付きディレクトリを検索
            pattern = f"{save_dir}_*"
            existing_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
            
            # 既存の番号を取得
            numbers = []
            for d in existing_dirs:
                try:
                    num_str = d.replace(save_dir + '_', '')
                    if num_str.isdigit():
                        numbers.append(int(num_str))
                except:
                    continue
            
            # 次の番号を決定
            next_num = max(numbers) + 1 if numbers else 1
            save_dir = f"{save_dir}_{next_num:03d}"
            logger.log(f"Save directory already exists. Using: {save_dir}")
        
        self.save_dir = save_dir

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist_util.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            
        # 最終出力: 保存されたモデルディレクトリ
        final_model_dir = os.path.join(self.save_dir, self.timestamp)
        total_steps = self.step + self.resume_step
        model_file = f"model{total_steps:06d}.pt"
        full_model_path = os.path.join(final_model_dir, model_file)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"Model directory: {final_model_dir}")
        print(f"Total training steps: {total_steps}")
        print(f"Latest model file: {model_file}")
        print(f"Full model path: {full_model_path}")
        print(f"{'='*60}")
        
        # シェルスクリプト用の環境変数形式で出力
        print(f"\n# Shell script variables:")
        print(f"TRAINED_MODEL_PATH='{full_model_path}'")
        print(f"MODEL_DIR='{final_model_dir}'")
        print(f"TOTAL_STEPS={total_steps}")
        print(f"MODEL_NAME='{self.timestamp}'")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            original_loss = loss.item()  # 元のloss値を記録

            # --- ここから追加: ODE_ML_Hybridモデルの中の GeneODEインスタンスを見つけて self.soft==True の時だけ正則化を加算 ---
            # DDP 対応: ddp_model.module があれば中身、なければそのまま
            model_ref = getattr(self.ddp_model, "module", self.ddp_model)

            # ODE_ML_Hybridクラスの中のself.ode_modelを呼び出す
            ode_ref = getattr(model_ref, "ode_model", None)
            reg_value = 0.0
            if ode_ref is not None and getattr(ode_ref, "soft", False) and self.ode_reg_lambda > 0:
                reg = ode_ref.off_mask_penalty(self.ode_reg_norm)
                reg_value = reg.item()  # 正則化項の値を記録
                loss = loss + self.ode_reg_lambda * reg
            # --- 追加ここまで ---
            
            total_loss = loss.item()  # 合計loss値を記録
            
            # 損失の詳細を一時的に保存（save()で使用）
            self._current_loss_info = {
                'step': self.step + self.resume_step,
                'original_loss': original_loss,
                'reg_value': reg_value,
                'reg_weighted': reg_value * self.ode_reg_lambda,
                'total_loss': total_loss,
                'ode_reg_lambda': self.ode_reg_lambda
            }




            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            # ======================================================
            # 🚨 NaN 検出まとめブロック（最後に一括チェック）
            # ======================================================
            has_nan = False

            def check_nan(name, tensor):
                nonlocal has_nan
                if tensor is None:
                    return
                if th.isnan(tensor).any() or th.isinf(tensor).any():
                    print(f"❌ {name} contains NaN or Inf")
                    has_nan = True

            # ① モデル出力関係
            if "loss" in locals():
                check_nan("loss", loss)
            if "losses" in locals() and isinstance(losses, dict):
                for k, v in losses.items():
                    check_nan(f"losses[{k}]", v)
            check_nan("weights", weights)

            # ② パラメータ＆勾配
            for name, p in self.model.named_parameters():
                check_nan(f"param {name}", p)
                if p.grad is not None:
                    check_nan(f"grad {name}", p.grad)

            if has_nan:
                print(f"🚨 NaN detected at step {self.step}! Halting for inspection.")
                import sys; sys.exit(1)
            # ======================================================


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        # 保存ディレクトリを事前に作成
        save_path = os.path.join(self.save_dir, self.timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.save_dir, self.timestamp, filename), "wb") as f:
                    th.save(state_dict, f)
        
        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist_util.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_dir, self.timestamp, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
            
            # 損失記録をCSVファイルに保存（パラメータ保存と同じタイミング）
            if self.save_loss_details and hasattr(self, '_current_loss_info'):
                import csv
                loss_file_path = os.path.join(self.save_dir, self.timestamp, "loss_details.csv")
                
                # ファイルが存在しない場合はヘッダーを書く
                write_header = not os.path.exists(loss_file_path)
                
                with open(loss_file_path, 'a', newline='') as csvfile:
                    fieldnames = ['step', 'original_loss', 'reg_value', 'reg_weighted', 'total_loss', 'ode_reg_lambda']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if write_header:
                        writer.writeheader()
                    
                    # 現在の損失情報を書き込み
                    writer.writerow(self._current_loss_info)
                
                logger.log(f"Saved loss record at step {self._current_loss_info['step']} to {loss_file_path}")

        dist_util.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv(key, values.mean().item())
        # logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv(f"{key}_q{quartile}", sub_loss)
            # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
