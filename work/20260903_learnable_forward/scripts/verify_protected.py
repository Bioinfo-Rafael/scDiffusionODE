#!/usr/bin/env python3
"""Verify the unchanged guided_diffusion core against a read-only manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import PROTECTED_MANIFEST, file_sha256, read_json  # noqa: E402


PROTECTED_FILES = (
    "guided_diffusion/gaussian_diffusion.py",
    "guided_diffusion/train_util.py",
    "guided_diffusion/respace.py",
    "guided_diffusion/script_util.py",
    "guided_diffusion/resample.py",
    "guided_diffusion/cell_model.py",
    "guided_diffusion/fp16_util.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _entries_from_payload(payload) -> dict[str, str]:
    """Accept a small number of explicit, human-readable manifest shapes."""

    candidate = payload
    if isinstance(payload, Mapping):
        if "files" in payload:
            candidate = payload["files"]
        elif "protected_files" in payload:
            candidate = payload["protected_files"]
    if isinstance(candidate, Mapping):
        entries = {str(path): str(digest).lower() for path, digest in candidate.items()}
    elif isinstance(candidate, list):
        entries = {}
        for item in candidate:
            if not isinstance(item, Mapping) or "path" not in item or "sha256" not in item:
                raise ValueError(
                    "manifest list entries must contain path and sha256"
                )
            entries[str(item["path"])] = str(item["sha256"]).lower()
    else:
        raise ValueError(
            "manifest must be a path-to-sha256 mapping or contain files/protected_files"
        )
    return entries


def verify_manifest(path: Path) -> dict:
    payload = read_json(path)
    entries = _entries_from_payload(payload)
    expected_paths = set(PROTECTED_FILES)
    observed_paths = set(entries)
    if observed_paths != expected_paths:
        missing = sorted(expected_paths - observed_paths)
        unexpected = sorted(observed_paths - expected_paths)
        raise ValueError(
            "protected manifest must contain exactly the seven agreed files; "
            f"missing={missing}, unexpected={unexpected}"
        )

    checks = {}
    for relative in PROTECTED_FILES:
        expected = entries[relative]
        if _SHA256.fullmatch(expected) is None:
            raise ValueError(f"invalid SHA-256 for {relative}: {expected!r}")
        file_path = (REPO_ROOT / relative).resolve()
        try:
            file_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"protected path escapes repository: {relative}") from exc
        actual = file_sha256(file_path) if file_path.is_file() else None
        checks[relative] = {
            "expected": expected,
            "actual": actual,
            "matched": actual == expected,
        }
    return {
        "ok": all(value["matched"] for value in checks.values()),
        "manifest": str(path.resolve()),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", default=str(PROTECTED_MANIFEST))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_manifest(Path(args.manifest).expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
