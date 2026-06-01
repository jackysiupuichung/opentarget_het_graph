"""
Stage A.5: coverage audit for the NCT terminal-date cache.

Reads `data/nct_terminal_dates.parquet` and the clinical_precedence evidence
parquets, and prints coverage broken down by:
  - all unique clinicalReportIds vs failure-bucket rows
  - terminal_date_source field (which date won)
  - shift (years) between studyStartDate (currently used) and terminal_date
    -- this is the leak that the fix removes
  - by clinicalStage (PHASE_1..4) for failure rows
  - by overall_status

No parser changes happen here. Use the numbers to decide drop policy.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

CLIN_PREC_GLOB = (
    "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidenceDated/"
    "sourceId=clinical_precedence/*.parquet"
)
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "nct_terminal_dates.parquet"

FAILURE_KEYWORDS = ("negative", "safety", "side effect", "sideeffect", "toxicity")


def is_failure(reasons) -> bool:
    if reasons is None:
        return False
    if isinstance(reasons, float):
        return False
    try:
        joined = " ".join(str(x) for x in reasons).lower()
    except TypeError:
        return False
    return any(k in joined for k in FAILURE_KEYWORDS)


def has_stop_reason(reasons) -> bool:
    if reasons is None:
        return False
    if isinstance(reasons, float):
        return False
    try:
        return len(reasons) > 0
    except TypeError:
        return bool(reasons)


def main(cache_path: Path) -> None:
    files = sorted(glob.glob(CLIN_PREC_GLOB))
    df = pd.concat(
        [
            pd.read_parquet(
                f,
                columns=[
                    "clinicalReportId",
                    "clinicalStage",
                    "trialStopReasonCategories",
                    "studyStartDate",
                ],
            )
            for f in files
        ],
        ignore_index=True,
    )
    df["nct_id"] = df["clinicalReportId"].astype(str).str.upper()
    df["is_nct"] = df["nct_id"].str.match(r"^NCT\d{8}$").fillna(False)
    df["has_stop_reason"] = df["trialStopReasonCategories"].apply(has_stop_reason)
    df["is_failure_row"] = df["trialStopReasonCategories"].apply(is_failure)

    cache = pd.read_parquet(cache_path)
    cache["nct_id"] = cache["nct_id"].astype(str).str.upper()
    print(f"cache rows: {len(cache)} | found=True: {cache['found'].sum()}")

    merged = df.merge(cache, on="nct_id", how="left")

    def pct(num, denom):
        return f"{num}/{denom} ({100 * num / max(denom, 1):.2f}%)"

    print("\n=== ROW-LEVEL COVERAGE (clinical_precedence) ===")
    n = len(merged)
    print(f"  total rows: {n}")
    print(f"  NCT-shaped id: {pct(merged['is_nct'].sum(), n)}")
    print(f"  cache hit (found=True): {pct((merged['found'] == True).sum(), n)}")
    print(
        "  cache hit with terminal_date: "
        f"{pct(merged['terminal_date'].notna().sum(), n)}"
    )

    print("\n=== UNIQUE-ID COVERAGE ===")
    uniq = df.drop_duplicates(subset=["nct_id"])
    u = len(uniq)
    uniq_merged = uniq.merge(cache, on="nct_id", how="left")
    print(f"  unique ids: {u}")
    print(f"  NCT-shaped: {pct(uniq['is_nct'].sum(), u)}")
    print(
        f"  cache found=True: {pct((uniq_merged['found'] == True).sum(), u)}"
    )
    print(
        "  cache hit with terminal_date: "
        f"{pct(uniq_merged['terminal_date'].notna().sum(), u)}"
    )

    print("\n=== FAILURE-BUCKET ROW COVERAGE ===")
    fail = merged[merged["is_failure_row"]]
    nf = len(fail)
    print(f"  failure-bucket rows: {nf}")
    print(
        "  cache hit with terminal_date: "
        f"{pct(fail['terminal_date'].notna().sum(), nf)}"
    )
    print(
        "  would be DROPPED (no terminal_date): "
        f"{pct(fail['terminal_date'].isna().sum(), nf)}"
    )

    print("\n  by clinicalStage:")
    grp = fail.groupby("clinicalStage").agg(
        n=("nct_id", "size"),
        with_date=("terminal_date", lambda s: s.notna().sum()),
    )
    grp["coverage_pct"] = 100 * grp["with_date"] / grp["n"]
    print(grp.to_string())

    print("\n=== STOP-REASON (any) ROW COVERAGE ===")
    sr = merged[merged["has_stop_reason"]]
    print(f"  stop-reason rows: {len(sr)}")
    print(
        "  cache hit with terminal_date: "
        f"{pct(sr['terminal_date'].notna().sum(), len(sr))}"
    )

    print("\n=== WHICH FIELD WINS (terminal_date_source) ===")
    src = (
        merged.loc[merged["terminal_date"].notna(), "terminal_date_source"]
        .value_counts(dropna=False)
    )
    print(src.to_string())

    print("\n=== FAILURE: source breakdown ===")
    fsrc = (
        fail.loc[fail["terminal_date"].notna(), "terminal_date_source"]
        .value_counts(dropna=False)
    )
    print(fsrc.to_string())

    print("\n=== OVERALL_STATUS distribution (failure rows) ===")
    print(fail["overall_status"].value_counts(dropna=False).head(15).to_string())

    print("\n=== TEMPORAL SHIFT (years): terminal_date - studyStartDate ===")
    fail = fail.copy()
    fail["start_year"] = pd.to_datetime(
        fail["studyStartDate"], errors="coerce"
    ).dt.year
    fail["terminal_year"] = pd.to_datetime(
        fail["terminal_date"], errors="coerce"
    ).dt.year
    fail["shift_years"] = fail["terminal_year"] - fail["start_year"]
    shift = fail["shift_years"].dropna()
    print(f"  rows with both dates: {len(shift)}")
    if len(shift):
        print(f"  median shift: {shift.median():.1f} years")
        print(f"  mean   shift: {shift.mean():.2f} years")
        print(f"  pct shifted >= 1y: {100 * (shift >= 1).mean():.1f}%")
        print(f"  pct shifted >= 3y: {100 * (shift >= 3).mean():.1f}%")
        print(f"  pct shifted >= 5y: {100 * (shift >= 5).mean():.1f}%")
        print("  shift distribution (years):")
        print(shift.value_counts().sort_index().to_string())

    summary = {
        "rows_total": int(n),
        "rows_failure": int(nf),
        "rows_failure_with_terminal_date": int(fail["terminal_date"].notna().sum()),
        "rows_failure_dropped_if_strict": int(fail["terminal_date"].isna().sum()),
        "unique_ids": int(u),
        "unique_ids_with_terminal_date": int(
            uniq_merged["terminal_date"].notna().sum()
        ),
    }
    out_json = cache_path.with_name("nct_terminal_dates_audit.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary to {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    main(args.cache)
