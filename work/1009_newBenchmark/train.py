"""Stage 1 training with original loss/optimizer/EMA, sparse minibatch input."""
from common import *
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from guided_diffusion.cell_datasets_loader import custom_collate_fn


class SparseCells(Dataset):
    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, i):
        return dense_rows(self.matrix[i:i+1])[0], {}


class Batches:
    """Same shuffle/drop_last/collation as load_data(preprocess=False).

    A dedicated generator plus epoch cursor makes CPU continuation reproducible.
    Only a minibatch is densified, rather than the full training matrix.
    """
    def __init__(self, a, batch_size, seed, state=None, *, preprocess=False):
        require(preprocess is False, 'preprocess must be False')
        require(len(a) >= batch_size, 'drop_last requires at least one full batch')
        self.generator = torch.Generator().manual_seed(seed)
        if state:
            self.generator.set_state(state['epoch_rng'])
        self.loader = DataLoader(SparseCells(a.X), batch_size=batch_size, shuffle=True,
                                 num_workers=0, drop_last=True, collate_fn=custom_collate_fn,
                                 generator=self.generator)
        self.reset()
        if state:
            for _ in range(state['offset']):
                next(self.iterator)
                self.offset += 1

    def reset(self):
        self.epoch_rng = self.generator.get_state().clone()
        self.iterator = iter(self.loader)
        self.offset = 0

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.reset()
            batch = next(self.iterator)
        self.offset += 1
        return batch[0]

    def state(self):
        return dict(epoch_rng=self.epoch_rng, offset=self.offset)


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def loss_plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for p in sorted((run / 'logs').glob('loss_*.csv')):
        with p.open() as f:
            rows.extend(list(csv.DictReader(f)))
    fig, ax = plt.subplots()
    for row in rows:
        row['step'] = int(row['step'])
    rows.sort(key=lambda r: r['step'])
    ax.plot([r['step'] for r in rows], [float(r['loss']) for r in rows])
    ax.set(xlabel='Optimizer step', ylabel='START_X MSE', title='CellUNet Stage 1')
    fig.savefig(run / 'figures/loss.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def train(c, *, target_device='cpu', steps=None, resume=None):
    run = paths(c); cfg = c['training']; dev = device(target_device)
    limit = cfg['total_steps'] if steps is None else steps
    require(1 <= limit <= cfg['total_steps'], 'steps must be 1..30000 (absolute stop step)')
    SOURCE.seed_all(cfg['seed'])
    if resume:
        model, ck, a, _ = load_checkpoint(c, resume, dev, 'raw')
        opt = MODELS.optimizer_for(model, cfg); opt.load_state_dict(ck['optimizer'])
        ema = {k: v.to(dev) for k, v in ck['ema'].items()}
        start = ck['step']
        batches = Batches(a, cfg['batch_size'], cfg['seed'], ck['batches'], preprocess=False)
        restore_rng(ck['rng'])
    else:
        require(not (run / 'checkpoints/latest.json').exists(), 'checkpoint exists; use --resume')
        a, _ = read_data(c)
        model = MODELS.build_model(cfg, ids(a.var_names)).to(dev)
        opt = MODELS.optimizer_for(model, cfg)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        start = 0
        batches = Batches(a, cfg['batch_size'], cfg['seed'], preprocess=False)
    require(start < limit, 'checkpoint already reached requested stop step')
    model.train(); d = diffusion(c); sampler = OBJECTIVES.timestep_sampler(cfg, d)
    meta = json.loads((run / 'data/metadata.json').read_text())
    write_json(run / 'effective_config.json', c)
    name = f'loss_{start:06d}_{uuid.uuid4().hex[:8]}.csv'
    with (run / 'logs' / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss', 'lr_used']); writer.writeheader()
        for index in range(start, limit):
            x = SOURCE.finite('batch', batches.next().to(dev))
            t, w = sampler.sample(len(x), dev)
            opt.zero_grad(set_to_none=True)
            loss, _ = OBJECTIVES.training_loss(model, d, x, t, w, cfg)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    SOURCE.finite('gradient', parameter.grad)
            lr_used = opt.param_groups[0]['lr']
            opt.step(); MODELS.update_ema(ema, model, float(cfg['ema_rate']))
            # Match source's post-update, zero-based annealing exactly.
            for group in opt.param_groups:
                group['lr'] = cfg['lr'] * (1 - index / cfg['lr_anneal_steps'])
            step = index + 1
            writer.writerow(dict(step=step, loss=float(loss.detach()), lr_used=lr_used)); f.flush()
            if step % cfg['log_interval'] == 0 or step == limit:
                print(f'Stage 1 step={step} loss={float(loss.detach()):.6g}', flush=True)
            if step % cfg['save_interval'] == 0 or step == limit:
                ckpath = run / 'checkpoints' / f'model{step:06d}.pt'
                require(not ckpath.exists(), f'refusing overwrite: {ckpath}')
                for value in model.state_dict().values():
                    SOURCE.finite('checkpoint', value)
                payload = dict(schema='cellunet_stage1_start_x_v1', step=step, training=cfg,
                    source_hashes=c['source_hashes'], raw=model.state_dict(), ema=ema,
                    optimizer=opt.state_dict(), rng=rng_state(), batches=batches.state(),
                    gene_ids=ids(a.var_names), cell_ids=ids(a.obs_names),
                    data_sha256=meta['training_sha256'], expression_scale=SCALE,
                    training_n_cells=a.n_obs, prediction='x_start', device=str(dev))
                tmp = ckpath.with_suffix('.tmp'); torch.save(payload, tmp); os.replace(tmp, ckpath)
                write_json(run / 'checkpoints/latest.json', dict(checkpoint=str(ckpath), step=step))
    loss_plot(run)
    return ckpath


def main():
    p = parser(__doc__); device_arg(p)
    p.add_argument('--steps', type=int, help='Absolute stop step for smoke/continuation; default 30000')
    p.add_argument('--resume', nargs='?', const='latest', help='Checkpoint path, or latest if omitted')
    args = p.parse_args(); c = config(args.config)
    resume = resolve_checkpoint(c, None if args.resume == 'latest' else args.resume) if args.resume else None
    print(train(c, target_device=args.device, steps=args.steps, resume=resume))

if __name__ == '__main__':
    cli(main)
