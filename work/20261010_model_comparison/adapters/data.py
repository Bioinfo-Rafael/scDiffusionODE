"""Exact 10/09 HVG preparation and sparse batches, with audited Mouse GRN IDs."""
from __future__ import annotations
import ast
import csv
from pathlib import Path
from ..common import *
from .legacy import load_script


def write_h5ad(a, path):
    path = confined(path)
    require(not path.exists(), f'refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp.h5ad')
    try:
        a.write_h5ad(tmp)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def dense_rows(x):
    import numpy as np
    from scipy import sparse
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def validate_training(path, meta, c):
    import anndata as ad
    import numpy as np
    require(sha256(path) == meta['training_sha256'], 'training SHA256 mismatch')
    a = ad.read_h5ad(path)
    require(a.shape == (c['expected_cells'], c['n_hvg']), 'training dimensions mismatch')
    require(ids(a.var_names) == meta['gene_ids'] and ids(a.obs_names) == meta['cell_ids'], 'cell/gene order mismatch')
    require(meta['expression_scale'] == SCALE and a.uns['stage1_data']['preprocess'] == False, 'linear scale provenance required')
    require(a.uns['stage1_data']['hvg_method'] == 'seurat_log1p_copy_exact_rank', 'HVG recipe differs')
    require(a.uns['stage1_data']['input_sha256'] == meta['input_sha256'], 'source provenance mismatch')
    for start in range(0, a.n_obs, 256):
        x = dense_rows(a.X[start:start+256])
        require(np.isfinite(x).all() and (x >= 0).all(), 'invalid expression')
    require(all(k in a.obs and not a.obs[k].isna().any() for k in ('celltype', 'stage')), 'annotations missing')
    ery = module('data_preparation.20261007.benchmark.config').ERYTHROID_CELLTYPES
    require(int(a.obs.celltype.isin(ery).sum()) == c['expected_erythroid'], 'erythroid count mismatch')
    return a


def prepare(c, *, data_dir=None):
    dest = confined(data_dir or HERE / 'runs/data'); dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / 'manifest.json'
    if manifest.exists():
        read_data(c, dest)
        return manifest
    source_run = root_path(c['legacy_run'])
    existing = source_run / 'data/training.h5ad'
    if existing.is_file():
        meta = read_json(source_run / 'data/metadata.json')
        a = validate_training(existing, meta, c)
        source_input = root_path(c['input'])
        require(source_input.is_file(), 'source MouseGastrulation.h5ad required for reuse verification')
        require(sha256(source_input) == meta['input_sha256'], 'source input changed')
        training = existing
    else:
        legacy_c = dict(c, runs=str(dest / 'prepared'), input=str(root_path(c['input'])))
        def paths(_):
            out = dest / 'prepared'; (out / 'data').mkdir(parents=True, exist_ok=True); return out
        def read_existing(_):
            m = read_json(paths(_) / 'data/metadata.json')
            return validate_training(paths(_) / 'data/training.h5ad', m, c), m
        legacy = load_script('prepare_data', paths=paths, read_data=read_existing,
                             write_json=write_json, write_h5ad=write_h5ad)
        training = legacy.prepare(legacy_c)
        meta = read_json(paths(c) / 'data/metadata.json')
        a = validate_training(training, meta, c)
    grn = audit_grns(a, dest, root_path(c['grn_source']))
    write_json(manifest, dict(training=str(training), metadata=meta, grn=grn,
        gene_ids_sha256=digest_ids(a.var_names), cell_ids_sha256=digest_ids(a.obs_names)))
    (dest / 'gene_ids.txt').write_text('\n'.join(ids(a.var_names)) + '\n')
    return manifest


def read_data(c, data_dir=None):
    folder = Path(data_dir or HERE / 'runs/data').resolve()
    m = read_json(folder / 'manifest.json')
    a = validate_training(m['training'], m['metadata'], c)
    grn = m['grn']
    require(sha256(grn['mapped_tsv']) == grn['mapped_sha256'], 'mapped GRN changed')
    require(sha256(grn['source']) == grn['source_sha256'], 'GRN source changed')
    meta = dict(m['metadata'], grn=grn)
    return a, meta


def mapped_edges(a, path):
    """Resolve exact IDs first, then unambiguous case-sensitive gene symbols."""
    import pandas as pd
    genes = ids(a.var_names); direct = set(genes)
    symbols = {}
    if 'gene_name' in a.var:
        for gene, symbol in zip(genes, a.var.gene_name):
            if pd.notna(symbol):
                symbols.setdefault(str(symbol), []).append(gene)
    def resolve(value):
        value = str(value)
        if value.startswith('{'):
            try: value = str(ast.literal_eval(value)['gene'])
            except (ValueError, SyntaxError, KeyError, TypeError): return None
        if value in direct: return value
        match = symbols.get(value, [])
        return match[0] if len(match) == 1 else None
    frame = pd.read_csv(path, sep='\t', dtype=str)
    cols = ('from', 'to') if {'from', 'to'} <= set(frame) else ('from_ensembl', 'to_ensembl')
    require(set(cols) <= set(frame), f'unsupported GRN columns: {path}')
    src = frame[cols[0]].map(resolve); dst = frame[cols[1]].map(resolve)
    valid = src.notna() & dst.notna()
    edges = sorted(set(zip(src[valid], dst[valid])))
    participating = {g for pair in edges for g in pair}
    report = dict(source=str(path), source_sha256=sha256(path), input_edges=len(frame),
        matched_rows=int(valid.sum()), unique_edges=len(edges),
        edge_match_fraction=float(valid.mean()) if len(frame) else 0.,
        source_match_fraction=float(src.notna().mean()) if len(frame) else 0.,
        target_match_fraction=float(dst.notna().mean()) if len(frame) else 0.,
        participating_genes=len(participating), gene_coverage=len(participating)/len(genes),
        ambiguous_symbols=sum(len(v)>1 for v in symbols.values()),
        mapping='exact ID then unique case-sensitive gene_name; no orthology/case guessing')
    return edges, report


def audit_grns(a, output, selected):
    reports, chosen = [], None
    paths = sorted(set((ROOT / 'external_data').glob('*.tsv')) | {Path(selected)})
    for path in paths:
        edges, report = mapped_edges(a, path)
        reports.append(report)
        if path.resolve() == Path(selected).resolve(): chosen = (edges, report)
    write_json(output / 'grn_audit.json', reports)
    require(chosen is not None, 'selected GRN absent')
    edges, report = chosen
    # Keep data preparation useful for A01 even if GRN mapping fails. Field construction stops.
    mapped = output / 'mouse_grn_gene_ids.tsv'
    with mapped.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t'); writer.writerow(['from','to']); writer.writerows(edges)
    return dict(report, mapped_tsv=str(mapped), mapped_sha256=sha256(mapped),
                mask_orientation='target_source for 8/30 and June MathMLP; source_target for old GeneODE')


def mask_for(genes, meta):
    require(meta['grn']['unique_edges'] > 0, 'GRN has zero mapped edges; soft constraint cannot be claimed')
    require(sha256(meta['grn']['mapped_tsv']) == meta['grn']['mapped_sha256'], 'mapped GRN changed')
    loader = module('ODE.ode_20260609_mathmlp').build_edge_mask
    return loader(genes, meta['grn']['mapped_tsv']).T.contiguous()
