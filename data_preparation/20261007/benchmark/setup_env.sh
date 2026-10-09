#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
ENV_DIR="$HERE/../data/benchmark/.venv-eval"
EXPECTED_PYTHON=3.12.14
EXPECTED_COMMIT=719aafa3aebe488bf557c484ca700b78a0b90c95
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if [[ ! -f "$HERE/vendor/VeloEV/veloev/__init__.py" ]]; then
  git -C "$ROOT" submodule update --init --depth 1 -- data_preparation/20261007/benchmark/vendor/VeloEV
fi
[[ "$(git -C "$HERE/vendor/VeloEV" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo 'VeloEV submodule is not at the pinned commit; refusing to change it automatically.' >&2; exit 1;
}
[[ -z "$(git -C "$HERE/vendor/VeloEV" status --porcelain --untracked-files=no)" ]] || {
  echo 'VeloEV source is modified.' >&2; exit 1;
}
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  [[ "$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')" == "$EXPECTED_PYTHON" ]] || {
    echo "Set PYTHON_BIN to Python $EXPECTED_PYTHON (see README)." >&2; exit 1;
  }
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
[[ "$("$PY" -c 'import platform; print(platform.python_version())')" == "$EXPECTED_PYTHON" ]] || {
  echo "Existing evaluation venv is not Python $EXPECTED_PYTHON; move it aside first." >&2; exit 1;
}
export PYTHONNOUSERSITE=1
unset PYTHONPATH
"$PY" -m pip install --disable-pip-version-check pip==25.3 setuptools==78.1.1 wheel==0.45.1
"$PY" -m pip install --disable-pip-version-check --no-build-isolation -r "$HERE/requirements.lock"
"$PY" -m pip check
# VeloEV is imported straight from the pinned submodule, avoiding a second installed copy.
PYTHONPATH="$HERE" "$PY" -c 'from common import check_vendor; check_vendor(); import veloev; print(veloev.__file__)'
"$PY" -m pip freeze --all > "$HERE/../data/benchmark/environment.freeze.txt"
echo "Evaluation Python: $PY"
