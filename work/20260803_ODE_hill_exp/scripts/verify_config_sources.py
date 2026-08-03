#!/usr/bin/env python3
"""Verify that the checked-in base config matches its recorded comparison."""

from __future__ import annotations

import json
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = json.loads((SUITE_ROOT / "configs" / "base.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (SUITE_ROOT / "configs" / "source_comparison.json").read_text(encoding="utf-8")
    )
    failures = []
    for row in comparison["values"]:
        key = row["key"]
        current = row["current"]
        if key not in base:
            failures.append({"key": key, "reason": "missing from base.json"})
        elif base[key] != current:
            failures.append({"key": key, "recorded": current, "actual": base[key]})
    result = {"ok": not failures, "failures": failures, "checked_keys": len(comparison["values"])}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
