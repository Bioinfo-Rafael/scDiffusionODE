"""Two optimizers; reuse the 20260830 forward/loss and FP32 master parameters.

The legacy FP16 master tensors mix both branches. FP16 is explicitly rejected,
not silently split by the flattened parameter position. The original wrapper
also has no convert_to_fp16. This experiment retains its FP32 default.
"""
import copy
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from common import legacy, inside, torch_load, write_json
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import UniformSampler

LegacyLoop = legacy('training.train_loop_20260830').TrainLoop20260830


def split_parameters(model):
    cell, ode = list(model.ml_model.parameters()), list(model.ode_model.parameters())
    a, b, all_ids = set(map(id, cell)), set(map(id, ode)), set(map(id, model.parameters()))
    if a & b or a | b != all_ids or len(a) != len(cell) or len(b) != len(ode):
        raise ValueError('Optimizer parameter sets must be disjoint and exhaustive')
    return cell, ode


def norm(tensors):
    values = [t.detach().float().square().sum() for t in tensors if t is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


class DualStep:
    """One MPT optimize() call performs both FP32 optimizer steps exactly once."""
    def __init__(self, cell, ode):
        self.cell, self.ode = cell, ode
    def step(self):
        self.cell.step(); self.ode.step()


class TrainLoop(LegacyLoop):
    def __init__(self, *, model, diffusion, data, config, run_dir, resume=False):
        c = config
        if c['use_fp16']:
            raise ValueError('FP16 flattened master parameters cannot be split; use_fp16=false required')
        if dist_util.get_world_size() != 1:
            raise ValueError('This reproducible experiment supports one process/GPU per condition')
        self.config, self.run_dir = copy.deepcopy(c), inside(run_dir)
        self.model, self.diffusion, self.data = model, diffusion, data
        self.ddp_model, self.use_ddp = model, False
        self.batch_size = self.microbatch = c['batch_size']
        self.lr, self.lr_anneal_steps = c['lr'], c['total_steps']
        self.cell_ode_reg_lambda_20260830 = c['cell_ode_reg_lambda_20260830']
        self.cell_ode_reg_schedule_20260830 = 'constant'
        self.ode_reg_lambda, self.ode_reg_norm = c['ode_reg_lambda'], c['ode_reg_norm']
        self.save_loss_details = True
        self.schedule_sampler = UniformSampler(diffusion)
        self.step = self.resume_step = 0
        self.global_batch = self.batch_size
        self.mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False)
        self.cell_params, self.ode_params = split_parameters(model)
        assert set(map(id,self.mp_trainer.master_params)) == set(map(id,self.cell_params+self.ode_params))
        self.opt = torch.optim.AdamW(self.cell_params, lr=c['lr'], weight_decay=c['weight_decay'])
        self.ode_opt = torch.optim.SGD(self.ode_params, lr=c['ode_lr'], momentum=0., weight_decay=c['ode_weight_decay'])
        self.dual_opt = DualStep(self.opt, self.ode_opt)
        self.ema_rate = [float(v) for v in str(c['ema_rate']).split(',')]
        self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in self.ema_rate]
        self.folder = self.run_dir / 'checkpoints/segment_000/model'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ode_norms = []
        self.hook = model.ode_model.register_forward_hook(self._capture_ode)
        if resume:
            self.restore(resume if isinstance(resume, (str,Path)) else None)
        self.model.train()

    def _capture_ode(self, module, args, output):
        self.ode_norms.append(float(output.detach().float().norm(dim=1).mean()))

    def _anneal_lr(self):
        # self.step is the number of COMPLETED updates, including after resume.
        fraction = 1. - self.step / self.config['total_steps']
        for opt, base in ((self.opt,self.config['lr']), (self.ode_opt,self.config['ode_lr'])):
            for group in opt.param_groups:
                group['lr'] = base * fraction

    def run_step(self, batch, cond):
        self.ode_norms.clear()
        self.forward_backward(batch, cond)  # unchanged default full-batch legacy loss
        cell_before = [p.detach().clone() for p in self.cell_params]
        ode_before = [p.detach().clone() for p in self.ode_params]
        row = dict(self._current_loss_components_20260830)
        row.update(cell_gradient_norm=norm(p.grad for p in self.cell_params),
                   ode_gradient_norm=norm(p.grad for p in self.ode_params),
                   cell_learning_rate=self.opt.param_groups[0]['lr'],
                   ode_learning_rate=self.ode_opt.param_groups[0]['lr'],
                   ode_output_norm=float(np.mean(self.ode_norms)))
        if not self.mp_trainer.optimize(self.dual_opt):
            raise RuntimeError('Unexpected skipped FP32 optimizer step')
        self._update_ema()
        row.update(cell_update_norm=norm(p-b for p,b in zip(self.cell_params,cell_before)),
                   ode_update_norm=norm(p-b for p,b in zip(self.ode_params,ode_before)))
        self.step += 1
        self._anneal_lr()
        row['measurement_source'] = 'training_actual_optimizer_step'
        row['data_indices_sha256'] = hashlib.sha256(self.data.last_indices.tobytes()).hexdigest()
        for name in ('training_metrics.csv','loss_components_20260830.csv'):
            path = self.folder / name
            exists = path.exists()
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not exists: writer.writeheader()
                writer.writerow(row)
        return row

    def run_loop(self, stop_after=None):
        try:
            if self.step == 0:
                self.save()  # true initial parameters, before the first update
            end = min(self.config['total_steps'], stop_after or self.config['total_steps'])
            while self.step < end:
                batch, cond = next(self.data)
                self.run_step(batch, cond)
                if self.step % self.config['log_interval'] == 0:
                    logger.logkv('completed_steps', self.step); logger.dumpkvs()
                if self.step % self.config['save_interval'] == 0:
                    self.save()
            self.save()
        finally:
            self.hook.remove()

    def save(self):
        def atomic(value, path):
            temp = path.with_suffix('.tmp')
            torch.save(value,temp); temp.replace(path)
        step = self.step
        atomic(self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params), self.folder/f'model{step:06d}.pt')
        for rate, params in zip(self.ema_rate,self.ema_params):
            atomic(self.mp_trainer.master_params_to_state_dict(params), self.folder/f'ema_{rate}_{step:06d}.pt')
        state = dict(format='dual_optimizer_v1', completed_steps=step, config=self.config,
                     model=self.model.state_dict(), cell_optimizer=self.opt.state_dict(),
                     ode_optimizer=self.ode_opt.state_dict(), ema=self.ema_params,
                     data=self.data.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     lg_loss_scale=self.mp_trainer.lg_loss_scale)
        path = self.folder / f'trainer{step:06d}.pt'
        atomic(state,path)
        write_json(self.run_dir/'checkpoints/latest.json',dict(completed_steps=step,trainer=str(path)))

    def restore(self, path=None):
        from common import read_json
        path = inside(path or read_json(self.run_dir/'checkpoints/latest.json')['trainer'])
        path.relative_to(self.run_dir)
        s = torch_load(path)
        latest = read_json(self.run_dir/'checkpoints/latest.json')
        if s['completed_steps'] != latest['completed_steps']:
            raise ValueError('Resume must use the latest committed checkpoint; use a new run for branching')
        if s['format'] != 'dual_optimizer_v1' or s['config'] != self.config:
            raise ValueError('Resume requires a matching dual-optimizer checkpoint/config')
        self.model.load_state_dict(s['model'], strict=True)
        self.opt.load_state_dict(s['cell_optimizer']); self.ode_opt.load_state_dict(s['ode_optimizer'])
        device = next(self.model.parameters()).device
        self.ema_params = [[p.to(device) for p in group] for group in s['ema']]
        self.step = s['completed_steps']
        self.data.load_state_dict(s['data'])
        self.mp_trainer.lg_loss_scale = s['lg_loss_scale']
        torch.set_rng_state(s['torch_rng'])
        if s['cuda_rng'] is not None:
            if not torch.cuda.is_available(): raise ValueError('CUDA resume requires CUDA')
            torch.cuda.set_rng_state_all(s['cuda_rng'])
        np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
        # Discard rows newer than the chosen committed checkpoint, before appending.
        for name in ('training_metrics.csv','loss_components_20260830.csv'):
            p = self.folder/name
            if p.exists():
                with p.open(newline='') as f:
                    reader = csv.DictReader(f); fields = reader.fieldnames
                    rows = [r for r in reader if int(r['training_step']) <= self.step]
                with p.open('w', newline='') as f:
                    w = csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
