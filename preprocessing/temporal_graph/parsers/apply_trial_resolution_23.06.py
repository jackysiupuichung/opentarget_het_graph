"""
Stage B (23.06 variant): mutate the 23.06 chembl parquets in place.

Dual-edge mode mirrors apply_trial_resolution.py (the 26.03 path), but adapted
to 23.06's schema:

  - NCT lives inside the `urls` struct list as
    https://clinicaltrials.gov/ct2/show/NCT########
    (not in a dedicated clinicalReportId column).
  - `clinicalStatus` exists natively (no need to backfill from CT.gov cache
    for the ongoing/completed distinction). Values are title-case strings:
        Completed, Recruiting, Terminated, Withdrawn, Suspended,
        Active, not recruiting, Not yet recruiting, Enrolling by invitation,
        Unknown status
  - Stop reasons come from `studyStopReasonCategories` (full-text strings like
    "Safety or side effects", "Negative", "Insufficient enrollment", etc.).

Bucketing per row:
  - has stop_reason 'safety'/'side effect'     -> adverse_effects
  - has stop_reason 'negative'                 -> unmet_efficacy
  - has stop_reason (anything else)            -> Unknown/Operational
  - clinicalStatus == 'Completed'              -> positive
  - clinicalStatus in {Recruiting, Active not recruiting,
                       Not yet recruiting, Enrolling by invitation}
                                                -> ongoing (no outcome row)
  - anything else (Terminated/Withdrawn/Suspended/Unknown/None)
                                                -> Unknown/Operational

Dual-edge emission:
  - Every NCT-shaped trial gets an `ongoing` row at studyStartDate.
  - Resolved trials (anything that produces an outcome bucket above) also get
    an outcome row at terminal_date (clipped to >= studyStartDate). Ongoing
    statuses contribute only the ongoing row.

Rows whose `urls` contains no NCT-shaped identifier are dropped (no terminal
date can be resolved). Idempotent: tolerates running on already-mutated
parquets.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

CHEMBL_DIR = Path(
    "/gpfs/scratch/bty414/opentarget_evidences/23.06/evidenceDated/sourceId=chembl"
)
DEFAULT_CACHE = (
    Path(__file__).resolve().parents[3] / "data" / "nct_terminal_dates.parquet"
)

NCT_RE = re.compile(r"(NCT\d{8})")

ONGOING_STATUSES = {
    "RECRUITING",
    "ACTIVE, NOT RECRUITING",
    "NOT YET RECRUITING",
    "ENROLLING BY INVITATION",
}
POSITIVE_STATUSES = {"COMPLETED"}

BUCKET_POS = "clinical_trial_positive"
BUCKET_NEG = "clinical_trial_unmet_efficacy"
BUCKET_ADV = "clinical_trial_adverse_effects"
BUCKET_OPS = "clinical_trial_Unknown/Operational"
BUCKET_ONG = "clinical_trial_ongoing"


def _has_stop_reason(v) -> bool:
    if v is None or isinstance(v, float):
        return False
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


def _failure_subbucket(reasons) -> str:
    if reasons is None or isinstance(reasons, float):
        return BUCKET_OPS
    try:
        joined = " ".join(str(r).strip().lower() for r in reasons)
    except TypeError:
        return BUCKET_OPS
    if "safety" in joined or "side effect" in joined or "sideeffect" in joined:
        return BUCKET_ADV
    if "negative" in joined:
        return BUCKET_NEG
    return BUCKET_OPS


def _extract_ncts(urls_val) -> list[str]:
    if urls_val is None or isinstance(urls_val, float):
        return []
    out = []
    try:
        for item in urls_val:
            if isinstance(item, dict):
                u = str(item.get("url") or "")
                m = NCT_RE.search(u)
                if m:
                    out.append(m.group(1))
    except TypeError:
        pass
    return out


def outcome_bucket(row) -> str | None:
    """Return outcome bucket if resolved, else None (still ongoing)."""
    if _has_stop_reason(row["studyStopReasonCategories"]):
        return _failure_subbucket(row["studyStopReasonCategories"])
    s = (row.get("clinicalStatus") or "")
    s = s.upper().strip() if isinstance(s, str) else ""
    if s in POSITIVE_STATUSES:
        return BUCKET_POS
    if s in ONGOING_STATUSES:
        return None
    # Terminated / Withdrawn / Suspended / Unknown / None -> operational
    return BUCKET_OPS


def _clip_to_start(term, start) -> str | None:
    if pd.isna(term) or term is None:
        return str(start) if pd.notna(start) else None
    if pd.isna(start) or start is None:
        return str(term)
    try:
        if pd.to_datetime(term, errors="coerce") < pd.to_datetime(start, errors="coerce"):
            return str(start)
    except Exception:
        pass
    return str(term)


def process_file(path: Path, cache: pd.DataFrame, dry_run: bool) -> dict:
    df = pd.read_parquet(path)
    n_before = len(df)

    # Drop prior Stage B mutations so we always start clean.
    df = df.drop(
        columns=[c for c in ("trial_outcome_bucket", "resolvedTrialDate") if c in df.columns]
    )

    # Extract first NCT per row from the urls struct.
    ncts = df["urls"].apply(_extract_ncts)
    df = df.assign(_nct_id=ncts.apply(lambda lst: lst[0].upper() if lst else None))
    has_nct = df["_nct_id"].notna()
    n_dropped = int((~has_nct).sum())
    df = df[has_nct].copy()

    # Merge terminal_date from cache (overall_status from cache is unused here
    # because 23.06 has clinicalStatus natively).
    df = df.merge(
        cache[["nct_id", "terminal_date"]].rename(columns={"nct_id": "_nct_id"}),
        on="_nct_id",
        how="left",
    )

    # 1) ongoing row for every NCT trial, dated at studyStartDate.
    ongoing = df.copy()
    ongoing["trial_outcome_bucket"] = BUCKET_ONG
    ongoing["resolvedTrialDate"] = ongoing["studyStartDate"].astype(object)
    ongoing.loc[ongoing["studyStartDate"].isna(), "resolvedTrialDate"] = None

    # 2) outcome row only for resolved trials, dated at terminal_date (clipped).
    df["_outcome"] = df.apply(outcome_bucket, axis=1)
    resolved = df[df["_outcome"].notna()].copy()
    resolved["trial_outcome_bucket"] = resolved["_outcome"]
    resolved["resolvedTrialDate"] = resolved.apply(
        lambda r: _clip_to_start(r.get("terminal_date"), r.get("studyStartDate")),
        axis=1,
    )
    resolved = resolved.drop(columns=["_outcome"])

    out = pd.concat([ongoing, resolved], ignore_index=True)
    out = out.drop(columns=["_nct_id", "terminal_date"])

    counts = out["trial_outcome_bucket"].value_counts().to_dict()

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        out.to_parquet(tmp, index=False)
        shutil.move(str(tmp), str(path))

    return {
        "file": path.name,
        "rows_before": n_before,
        "rows_dropped_no_nct": n_dropped,
        "rows_after": int(len(out)),
        "buckets": counts,
    }


def main(cache_path: Path, dry_run: bool) -> None:
    cache = pd.read_parquet(cache_path)
    cache["nct_id"] = cache["nct_id"].astype(str).str.upper()

    files = sorted(CHEMBL_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files at {CHEMBL_DIR}")
    print(f"{'DRY-RUN: ' if dry_run else ''}processing {len(files)} files")

    totals = {
        "rows_before": 0,
        "rows_dropped_no_nct": 0,
        "rows_after": 0,
        "buckets": {},
    }
    for path in files:
        r = process_file(path, cache, dry_run)
        print(
            f"  {r['file']}: {r['rows_before']} -> {r['rows_after']} "
            f"(dropped no-NCT {r['rows_dropped_no_nct']})"
        )
        totals["rows_before"] += r["rows_before"]
        totals["rows_dropped_no_nct"] += r["rows_dropped_no_nct"]
        totals["rows_after"] += r["rows_after"]
        for k, v in r["buckets"].items():
            totals["buckets"][k] = totals["buckets"].get(k, 0) + int(v)

    print("\n=== TOTALS ===")
    print(f"  rows before:           {totals['rows_before']}")
    print(f"  rows dropped (no NCT): {totals['rows_dropped_no_nct']}")
    print(f"  rows after:            {totals['rows_after']}")
    print("\n  bucket distribution:")
    for k in sorted(totals["buckets"], key=lambda x: -totals["buckets"][x]):
        print(f"    {k:42s} {totals['buckets'][k]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.cache, args.dry_run)
