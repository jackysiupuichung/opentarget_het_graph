"""
Stage A: build an offline cache of NCT terminal dates from ClinicalTrials.gov v2.

Collects NCT IDs from two Open Targets evidence dumps:
  - 26.03 sourceId=clinical_precedence: NCT lives in `clinicalReportId`
  - 23.06 sourceId=chembl:               NCT lives inside `urls` struct list as
                                         https://clinicaltrials.gov/ct2/show/NCT########

For every distinct NCT ID, fetch the StatusModule and resolve a single
`terminal_date` using the priority:
    resultsFirstPostDate > lastUpdatePostDate > completionDate(ACTUAL)
    > primaryCompletionDate(ACTUAL) > completionDate(ESTIMATED)
    > primaryCompletionDate(ESTIMATED)

Non-NCT IDs and IDs not returned by the API are written with found=False so
coverage is explicit downstream. Resumable: rerun and only missing IDs are
fetched.
"""

from __future__ import annotations

import argparse
import glob
import re
import time
from pathlib import Path

import pandas as pd
import requests

CLIN_PREC_GLOB = (
    "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidenceDated/"
    "sourceId=clinical_precedence/*.parquet"
)
CHEMBL_2306_GLOB = (
    "/gpfs/scratch/bty414/opentarget_evidences/23.06/evidenceDated/"
    "sourceId=chembl/*.parquet"
)
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "nct_terminal_dates.parquet"

CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
BATCH_SIZE = 50
SLEEP_SEC = 0.3
NCT_RE = re.compile(r"^NCT\d{8}$")

DATE_PRIORITY = [
    ("resultsFirstPostDateStruct", None),
    ("lastUpdatePostDateStruct", None),
    ("completionDateStruct", "ACTUAL"),
    ("primaryCompletionDateStruct", "ACTUAL"),
    ("completionDateStruct", "ESTIMATED"),
    ("primaryCompletionDateStruct", "ESTIMATED"),
]


_NCT_IN_URL_RE = re.compile(r"(NCT\d{8})")


def _collect_from_clinical_precedence() -> tuple[list[str], list[str]]:
    """26.03 path: NCT is in clinicalReportId."""
    files = sorted(glob.glob(CLIN_PREC_GLOB))
    if not files:
        return [], []
    df = pd.concat(
        [pd.read_parquet(f, columns=["clinicalReportId"]) for f in files],
        ignore_index=True,
    )
    ids = df["clinicalReportId"].dropna().astype(str).str.upper().unique().tolist()
    nct = [x for x in ids if NCT_RE.match(x)]
    non_nct = [x for x in ids if not NCT_RE.match(x)]
    return nct, non_nct


def _extract_ncts_from_urls(urls_val) -> list[str]:
    if urls_val is None:
        return []
    if isinstance(urls_val, float):
        return []
    out = []
    try:
        for item in urls_val:
            if isinstance(item, dict):
                u = str(item.get("url") or "")
                m = _NCT_IN_URL_RE.search(u)
                if m:
                    out.append(m.group(1))
    except TypeError:
        pass
    return out


def _collect_from_chembl_2306() -> tuple[list[str], list[str]]:
    """23.06 path: NCT is inside the `urls` struct list as
    https://clinicaltrials.gov/ct2/show/NCT########.
    """
    files = sorted(glob.glob(CHEMBL_2306_GLOB))
    if not files:
        return [], []
    dfs = [pd.read_parquet(f, columns=["urls"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    nct = set()
    for v in df["urls"]:
        for x in _extract_ncts_from_urls(v):
            nct.add(x.upper())
    return sorted(nct), []


def collect_input_ids() -> tuple[list[str], list[str]]:
    """Aggregate NCTs from both 26.03 clinical_precedence and 23.06 chembl.

    Non-NCT identifiers (EudraCT etc.) only come from the 26.03 dump (the
    23.06 urls field has only NCTs).
    """
    nct_a, non_nct = _collect_from_clinical_precedence()
    nct_b, _ = _collect_from_chembl_2306()
    nct = sorted(set(nct_a) | set(nct_b))
    non_nct = sorted(set(non_nct))
    return nct, non_nct


def resolve_terminal(status: dict) -> tuple[str | None, str | None]:
    for field, required_type in DATE_PRIORITY:
        blob = status.get(field)
        if not blob:
            continue
        if required_type and blob.get("type") != required_type:
            continue
        date = blob.get("date")
        if date:
            label = field.replace("Struct", "")
            if required_type:
                label = f"{label}({required_type})"
            return date, label
    return None, None


def fetch_batch(ids: list[str], session: requests.Session) -> dict[str, dict]:
    params = {
        "filter.ids": "|".join(ids),
        "pageSize": len(ids),
        "fields": (
            "protocolSection.identificationModule.nctId,"
            "protocolSection.statusModule"
        ),
        "format": "json",
    }
    r = session.get(CTGOV_URL, params=params, timeout=60)
    r.raise_for_status()
    out = {}
    for study in r.json().get("studies", []):
        proto = study.get("protocolSection", {})
        nct = proto.get("identificationModule", {}).get("nctId")
        if nct:
            out[nct.upper()] = proto.get("statusModule", {})
    return out


def build_cache(out_path: Path, limit: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nct_ids, non_nct = collect_input_ids()
    print(f"input: {len(nct_ids)} NCT-shaped, {len(non_nct)} non-NCT")

    existing: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        existing = {row["nct_id"]: row.to_dict() for _, row in prev.iterrows()}
        print(f"existing cache: {len(existing)} rows")

    todo = [x for x in nct_ids if x not in existing]
    if limit:
        todo = todo[:limit]
    print(f"to fetch: {len(todo)}")

    session = requests.Session()
    new_rows: list[dict] = []
    t0 = time.time()
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        try:
            results = fetch_batch(batch, session)
        except Exception as exc:
            print(f"batch {i // BATCH_SIZE} failed: {exc}; retrying once")
            time.sleep(2.0)
            try:
                results = fetch_batch(batch, session)
            except Exception as exc2:
                print(f"batch {i // BATCH_SIZE} failed again: {exc2}; marking missing")
                results = {}

        for nct in batch:
            status = results.get(nct)
            if status is None:
                new_rows.append(
                    {
                        "nct_id": nct,
                        "found": False,
                        "terminal_date": None,
                        "terminal_date_source": None,
                        "overall_status": None,
                        "why_stopped": None,
                    }
                )
                continue
            date, source = resolve_terminal(status)
            new_rows.append(
                {
                    "nct_id": nct,
                    "found": True,
                    "terminal_date": date,
                    "terminal_date_source": source,
                    "overall_status": status.get("overallStatus"),
                    "why_stopped": status.get("whyStopped"),
                }
            )

        time.sleep(SLEEP_SEC)
        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / max(elapsed, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            print(f"  {done}/{len(todo)} ({rate:.1f}/s, eta {eta/60:.1f}m)")

    # Add non-NCT IDs as found=False (only on initial build)
    for x in non_nct:
        if x not in existing and not any(r["nct_id"] == x for r in new_rows):
            new_rows.append(
                {
                    "nct_id": x,
                    "found": False,
                    "terminal_date": None,
                    "terminal_date_source": None,
                    "overall_status": None,
                    "why_stopped": "non-NCT identifier",
                }
            )

    combined = list(existing.values()) + new_rows
    out = pd.DataFrame(combined).drop_duplicates(subset=["nct_id"], keep="last")
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="dev: cap fetches")
    args = ap.parse_args()
    build_cache(args.out, limit=args.limit)
