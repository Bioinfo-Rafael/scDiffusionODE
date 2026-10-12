"""Exact-hash, D-only compatibility for the reviewed model-expansion change."""
from ..common import HERE, read_json, require, source_hashes


def verify_sources(saved, config, current=None):
    current=source_hashes() if current is None else current
    changed={p for p in set(saved)|set(current) if saved.get(p)!=current.get(p)}
    if not changed:return []
    require(config['family']=='consistency' and config['ode']=='hill_after_linear'
            and config.get('ode_optimizer')=='SGD' and not config.get('exp_clip'),
            'source/config code changed since checkpoint')
    rules=read_json(HERE/'audit/d_sgd_source_compatibility.json')['files']
    rejected=[p for p in sorted(changed) if p not in rules or
              saved.get(p) not in rules[p]['saved'] or current.get(p)!=rules[p]['current']]
    require(not rejected,'unreviewed source changes since checkpoint: '+', '.join(rejected))
    return sorted(changed)
