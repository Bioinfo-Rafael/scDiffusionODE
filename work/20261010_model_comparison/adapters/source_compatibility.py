"""Exact-hash compatibility for reviewed changes; numerical sources stay strict."""
from ..common import HERE, expansion_conditions, read_json, require, source_hashes


def verify_sources(saved, config, current=None):
    current=source_hashes() if current is None else current
    changed={p for p in set(saved)|set(current) if saved.get(p)!=current.get(p)}
    if not changed:return []
    details='; '.join(f'{p}: saved={saved.get(p)} current={current.get(p)}' for p in sorted(changed))
    if config.get('condition') in expansion_conditions():
        filename='model_expansion_source_compatibility.json'
    elif (config['family']=='consistency' and config['ode']=='hill_after_linear'
          and config.get('ode_optimizer')=='SGD' and not config.get('exp_clip')):
        filename='d_sgd_source_compatibility.json'
    else:
        raise ValueError('source/config code changed since checkpoint: '+details)
    rules=read_json(HERE/'audit'/filename)['files']
    rejected=[p for p in sorted(changed) if p not in rules or
              saved.get(p) not in rules[p]['saved'] or current.get(p)!=rules[p]['current']]
    require(not rejected,'unreviewed source changes since checkpoint: '+details)
    return sorted(changed)
