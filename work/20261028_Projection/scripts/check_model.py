"""Read-only D20 check. A missing checkpoint is an error, never a synthetic replacement."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common.velocity_adapter import load_d20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE/'config.json'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    velocity, adata = load_d20(config, args.device)
    x = adata.X[:8]
    x = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    v = velocity(x)
    assert v.shape == x.shape and np.isfinite(v).all()
    np.testing.assert_allclose(velocity(x[0]), v[0], rtol=1e-5, atol=1e-6)
    print(json.dumps(dict(shape=list(adata.shape), velocity_norms=np.linalg.norm(v, axis=1).tolist(),
                          provenance=velocity.provenance), indent=2))


if __name__ == '__main__':
    main()
