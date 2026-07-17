#!/usr/bin/env python3
"""Export the 26.03 w3 clinical-advancement pipeline labels as train/test CSVs
for build_event_graph.py (columns: target_id, disease_id, transition_year, outcome).

train_dataset.csv  <- datasets_26.03_w3/training_dataset.zarr
test_dataset.csv   <- datasets_26.03_w3/evaluation_dataset.zarr
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path("/gpfs/scratch/bty414/clinical_advancement_paper/data/datasets_26.03_w3")
OUT = Path("/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/"
           "data/clinical_trial_advancement/26.03_w3")
OUT.mkdir(parents=True, exist_ok=True)


def export(zarr_name, out_csv):
    ds = xr.open_zarr(SRC / zarr_name).load()
    tid = [str(x) for x in ds.coords["target_id"].values]
    did = [str(x) for x in ds.coords["disease_id"].values]
    outcome = np.asarray(ds.outcome.values).astype(float).ravel().astype(bool)
    # transition_year lives in the descriptor coord
    dd = list(ds.coords["descriptors"].values)
    ty = np.asarray(ds.descriptor.values)[:, dd.index("transition_year")].astype(int)
    df = pd.DataFrame({
        "target_id": tid,
        "disease_id": did,
        "transition_year": ty,
        "outcome": outcome,
    })
    df.to_csv(out_csv, index=False)
    print(f"{out_csv.name}: {len(df)} rows, pos={int(outcome.sum())} "
          f"({outcome.mean()*100:.1f}%), years {ty.min()}-{ty.max()}")


export("training_dataset.zarr", OUT / "train_dataset.csv")
export("evaluation_dataset.zarr", OUT / "test_dataset.csv")
print(f"\nWrote to {OUT}")
