"""Strict D20 checkpoint/real-cell boundary for the existing noeqthermo pipeline."""
from __future__ import annotations
import importlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
THERMO = ROOT / 'work/20260821_noeqThermo'
for folder in (ROOT, THERMO / 'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
P = 'work.20261010_model_comparison'
C = importlib.import_module(P + '.common')
K = importlib.import_module(P + '.adapters.checkpoint')
DATA = importlib.import_module(P + '.adapters.data')
V = importlib.import_module(P + '.adapters.velocity')
MODELS = importlib.import_module(P + '.adapters.model')
CONDITION = 'D20_lora_consistency_lambda1'
DEFAULT_RUN = C.HERE / 'runs/model_expansion_20261012' / CONDITION


def load_analysis_config(mode, path=None):
    from noeqthermo.config import load_config
    c = load_config(mode, path or HERE / 'configs' / (mode + '.json'))
    C.require(c['sde']['parameterization'] == 'paper_literal' and
              c['sde']['D'] == c['sde']['dt'] == 0.2, 'D20 requires paper_literal D=dt=0.2')
    C.require(c['gene_id_source'] == 'var_names', 'D20 requires ordered var_names gene IDs')
    lap = c['lap']
    C.require(lap['D_scale'] == 1 and lap['bidirectional_pairs'],
              'D20 LAP requires bidirectional states and D_scale=1')
    C.require(lap['max_pairs'] % 2 == 0, 'LAP max_pairs must be even')
    labels = lap.get('endpoint_labels', [])
    C.require(not labels or (len(labels) >= 2 and len(set(labels)) == len(labels)),
              'LAP needs at least two distinct explicit labels')
    return c


def annotation_report(a):
    import pandas as pd
    return {key: {str(k): int(v) for k, v in a.obs[key].value_counts(dropna=False).items()}
            for key in a.obs if isinstance(a.obs[key].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(a.obs[key])}


class D20Adapter:
    def __init__(self, run=DEFAULT_RUN, *, data_dir=None, checkpoint=None,
                 weights='ema', device='auto', velocity_export=None):
        self.run = C.root_path(run)
        C.require(weights in ('ema', 'raw'), 'weights must be ema or raw')
        self.weights = weights
        self.config = C.read_json(self.run / 'effective_config.json')
        c = self.config
        C.validate(c)
        C.require(c['condition'] == CONDITION and c['family'] == 'consistency' and
                  c['ode'] == 'lora' and c['hybrid'] == 'none' and
                  c['consistency_lambda'] == 1.0 and c['field_timestep'] == 49,
                  'requires D20 LoRA consistency lambda1 ODE branch at t49')
        self.data_dir = C.root_path(data_dir) if data_dir else C.HERE / 'runs/data'
        self.adata, self.meta = DATA.read_data(c, self.data_dir)
        self.data_path = Path(C.read_json(self.data_dir / 'manifest.json')['training']).resolve()
        # This is the original loader: config, IDs/order, data, source and GRN
        # checks remain mandatory, even when an existing velocity export is used.
        self.model, ck, self.checkpoint = K.load(c, self.run, self.meta,
            value=checkpoint, weights=weights, target=C.device(device))
        C.require(ck['step'] == c['total_steps'], 'checkpoint must be the final training step')
        if not checkpoint or checkpoint == 'latest':
            pointer = C.read_json(self.run / 'checkpoints/latest.json')
            C.require(pointer['step'] == ck['step'], 'latest.json step mismatch')
        C.require(isinstance(self.model.ode_model, MODELS.math_fields.LoRAField),
                  'loaded ODE branch is not the original LoRAField')
        self.stamp = dict(condition=CONDITION, branch='ode_raw', field_source='ode_raw',
            definition=V.definition(c, 'ode_raw'), diffusion_timestep=49, weights=weights,
            checkpoint_path=str(self.checkpoint), checkpoint_sha256=C.sha256(self.checkpoint),
            checkpoint_step=ck['step'], input_expression_scale=C.SCALE,
            config_sha256=hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(),
            input_policy='clean X; no q_sample, subtraction, integration or output rescaling',
            training_sha256=self.meta['training_sha256'], input_sha256=self.meta['input_sha256'],
            cell_ids_sha256=C.digest_ids(self.meta['cell_ids']), gene_ids_sha256=C.digest_ids(self.meta['gene_ids']),
            model_class=type(self.model).__module__ + '.' + type(self.model).__name__,
            ode_class=type(self.model.ode_model).__module__ + '.' + type(self.model.ode_model).__name__,
            effective_config=c, training_source_hashes=ck['provenance']['source_hashes'],
            available_labels=annotation_report(self.adata),
            biological_assumption='experimental proxy ds/dt := raw output; consistency/START_X training does not establish biological ds/dt')
        candidate = C.root_path(velocity_export) if velocity_export else self.run / 'predictions/ode_raw_t49.h5ad'
        self.cached = None
        if velocity_export or (weights == 'ema' and candidate.exists()):
            self._load_export(candidate)
        self.stamp['velocity_origin'] = str(candidate) if self.cached is not None else 'checkpoint inference on selected real cells'

    def _load_export(self, path):
        import anndata as ad
        sidecar = C.read_json(path.with_suffix('.json'))
        for key in ('condition', 'branch', 'field_source', 'definition', 'diffusion_timestep',
                    'weights', 'checkpoint_sha256', 'input_expression_scale', 'input_policy',
                    'training_sha256', 'input_sha256', 'cell_ids_sha256', 'gene_ids_sha256', 'model_class'):
            C.require(sidecar.get(key) == self.stamp[key], 'velocity provenance mismatch: ' + key)
        C.require(sidecar.get('cell_ids') == self.meta['cell_ids'] and
                  sidecar.get('gene_ids') == self.meta['gene_ids'], 'export cell/gene IDs/order mismatch')
        digest = C.sha256(path)
        C.require(sidecar.get('output_sha256') == digest, 'velocity artifact SHA256 mismatch')
        a = ad.read_h5ad(path)
        C.require(a.obs_names.tolist() == self.meta['cell_ids'] and
                  a.var_names.tolist() == self.meta['gene_ids'], 'velocity h5ad ID order mismatch')
        C.require('velocity' in a.layers and a.layers['velocity'].shape == self.adata.shape,
                  'velocity export shape mismatch')
        self.cached = a.layers['velocity']
        self.stamp['velocity_export_sha256'] = digest

    def __call__(self, x, genes, inputs, device=None):
        C.require(list(genes) == self.meta['gene_ids'], 'evaluation gene IDs/order mismatch')
        cell_ids = inputs['evaluation_cell_ids']
        C.require(len(cell_ids) == len(set(cell_ids)), 'duplicate evaluation cell IDs')
        rows = self.adata.obs_names.get_indexer(cell_ids)
        C.require((rows >= 0).all() and x.shape == (len(rows), len(genes)), 'evaluation cell IDs/shape mismatch')
        out = np.empty(x.shape, dtype=np.float32)
        dev = next(self.model.parameters()).device
        batch = self.config['inference_batch_size']
        with torch.no_grad():
            for start in range(0, len(x), batch):
                sl = slice(start, start + batch)
                real = DATA.dense_rows(self.adata.X[rows[sl]])
                C.require(np.array_equal(x[sl], real), 'input is not the ordered real-cell clean X')
                if self.cached is not None:
                    out[sl] = DATA.dense_rows(self.cached[rows[sl]])
                else:
                    xt = torch.from_numpy(real).to(dev)
                    t = torch.full((len(xt), 1), 49, dtype=torch.long, device=dev)
                    # Exact existing export branch call. No CellUNet/Hybrid forward.
                    v = self.model.ode_model(xt, t)
                    C.require(tuple(v.shape) == tuple(xt.shape), 'ODE velocity shape mismatch')
                    out[sl] = v.cpu().float().numpy()
        C.require(np.isfinite(out).all(), 'nonfinite gene velocity')
        norms = np.linalg.norm(out, axis=1)
        diagnostics = dict(shape=list(out.shape), nonfinite_count=0,
            near_zero_fraction=float(np.mean(norms < 1e-8)),
            norm_quantiles=np.quantile(norms, [0, .25, .5, .75, 1]).tolist())
        return out, dict(self.stamp, diagnostics=diagnostics)

    def identity(self, config):
        import importlib.metadata as md
        files = list((HERE / 'src').glob('*.py')) + list((HERE / 'scripts').glob('*.py'))
        files += list((THERMO / 'src/noeqthermo').glob('*.py'))
        packages = {p: md.version(p) for p in ('torch', 'numpy', 'scipy', 'scanpy', 'scvelo', 'dynamo-release', 'umap-learn')}
        return dict(schema='D20_noeq_v1', checkpoint=C.sha256(self.checkpoint), weights=self.weights,
            training=self.meta['training_sha256'], config={k: v for k, v in config.items() if not k.startswith('_')},
            gene_ids=C.digest_ids(self.meta['gene_ids']), cell_ids=C.digest_ids(self.meta['cell_ids']),
            export=self.stamp.get('velocity_export_sha256'), device=str(next(self.model.parameters()).device),
            code={str(p.relative_to(ROOT)): C.sha256(p) for p in files},
            model_sources=C.source_hashes(), packages=packages)


def run(adapter, config, output):
    from noeqthermo.pipeline import run_pipeline, _write_json
    from noeqthermo.dynamo_bridge import import_dynamo
    # Dynamo imports matplotlib styling; import before scVelo's quiver-based
    # velocity autoscaling so first execution and resume use the same scale.
    import_dynamo(config['dynamo']['version'])
    import random
    # scVelo transition sampling and Dynamo topography use global RNGs.
    # Reset them before reconstruction so strict simulation fingerprints resume.
    random.seed(int(config['seed']))
    np.random.seed(int(config['seed']))
    torch.manual_seed(int(config['seed']))
    output = Path(output).expanduser().resolve()
    inputs = [dict(name=CONDITION, plot_label='D20 ODE: ' + adapter.weights + ', t=49',
        run_dir=adapter.run, checkpoint=adapter.checkpoint, data_path=adapter.data_path, config=adapter.config)]
    result = run_pipeline([adapter.run], config, output,
        device_name=str(next(adapter.model.parameters()).device), prepared_inputs=inputs,
        velocity_provider=adapter, prepared_adata=adapter.adata,
        resume_identity=adapter.identity(config))
    try:
        validate_outputs(output)
    except Exception as exc:
        result.update(status='failed_validation', error=f'{type(exc).__name__}: {exc}')
        _write_json(output / 'analysis_manifest.json', result)
        raise
    return result


def validate_outputs(output):
    from noeqthermo.pipeline import _write_json
    model_dir = output / 'models' / CONDITION
    with np.load(model_dir / 'landscape_flux_arrays.npz') as z:
        probability = z['probability_plot']
        C.require(np.isfinite(probability).all() and (probability >= 0).all() and
                  np.isclose(probability.sum(), 1), 'stationary probability normalization failed')
        visited = probability > 0
        C.require(np.allclose(z['potential_plot'][visited], -np.log(probability[visited])), 'potential identity failed')
        C.require(np.isinf(z['potential_plot'][~visited]).all(), 'unvisited potential must remain infinite')
        for key in ('gradient_force_x', 'gradient_force_y', 'probability_flux_x', 'probability_flux_y',
                    'rotational_force_x', 'rotational_force_y', 'curl_umap'):
            C.require(np.isfinite(z[key]).all(), 'nonfinite landscape quantity: ' + key)
        residual = np.hypot(z['mean_force_x_plot'] - z['gradient_force_x'] - z['rotational_force_x'],
                            z['mean_force_y_plot'] - z['gradient_force_y'] - z['rotational_force_y'])
        checks = dict(probability_sum=float(probability.sum()), visited_fraction=float(visited.mean()),
            finite_quantities=True, potential_identity=True,
            decomposition_residual_rms=float(np.sqrt(np.mean(residual[visited] ** 2))),
            note='Finite-difference log/gradient chain rule is approximate; residual is diagnostic, not a zero-tolerance test. Smoke does not establish stationarity.')
    lap = C.read_json(model_dir / 'least_action_metadata.json')['status']
    checks['lap'] = lap
    _write_json(output / 'numerical_validation.json', checks)
    if lap.get('failures') or lap.get('status') == 'failed':
        raise RuntimeError('LAP pair(s) failed; inspect least_action_metadata.json')
    return checks
