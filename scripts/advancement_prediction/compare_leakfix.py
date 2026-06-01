#!/usr/bin/env python3
"""Side-by-side comparison of the rev_advancement-leak fix.

Reads results.yaml from the leaked production run (p3_eahgt_both_lambdarank_v2)
and the leak-fixed re-run (p3_eahgt_both_lambdarank_leakfix), prints test
metrics, and flags any large delta (≥ 0.02 absolute on AUC/AP-style metrics)
as evidence the leak was material.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

_DEFAULT_LEAKED = "/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/p3_eahgt_both_lambdarank_v2"
_DEFAULT_FIXED = "/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/p3_eahgt_both_lambdarank_leakfix"

# Metrics we expect to see in results.yaml (subset; ignores missing ones).
_KEY_METRICS = [
    "test_roc_auc",
    "test_average_precision",
    "test_precision@10",
    "test_precision@30",
    "test_precision@50",
    "test_recall@10",
    "test_recall@30",
    "test_recall@50",
    "test_rs@10",
    "test_rs@30",
    "test_rs@50",
    "test_rs@100",
    "test_f1",
    "test_mcc",
]


def _load(path: Path) -> dict:
    cfg = OmegaConf.load(path / "results.yaml")
    flat = OmegaConf.to_container(cfg, resolve=True)
    test = flat.get("test", flat)
    return dict(test)


def main(leaked_dir: Path, fixed_dir: Path) -> None:
    leaked = _load(leaked_dir)
    fixed = _load(fixed_dir)

    print(f"{'metric':<32} {'leaked':>10} {'fixed':>10} {'delta':>10}")
    print("-" * 64)
    materially_changed = []
    for k in _KEY_METRICS:
        if k not in leaked and k not in fixed:
            continue
        a = leaked.get(k, float("nan"))
        b = fixed.get(k, float("nan"))
        d = (b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else float("nan")
        marker = ""
        if isinstance(d, float) and abs(d) >= 0.02:
            marker = "  ←"
            materially_changed.append((k, d))
        print(f"{k:<32} {a:>10.4f} {b:>10.4f} {d:>10.4f}{marker}")

    print()
    if materially_changed:
        print(f"⚠ {len(materially_changed)} metric(s) moved by ≥ 0.02 — leak was likely material.")
        print("  Recommend re-running every production checkpoint in DEFAULT_RUNS.")
    else:
        print("✓ No metric moved by ≥ 0.02. Leak was likely not material to reported metrics.")
        print("  Footnote in paper may be sufficient; full re-run probably not required.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--leaked", default=_DEFAULT_LEAKED, type=Path)
    p.add_argument("--fixed", default=_DEFAULT_FIXED, type=Path)
    args = p.parse_args()
    main(args.leaked, args.fixed)
