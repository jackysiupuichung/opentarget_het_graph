# THBKG

The **Temporal Heterogeneous Biomedical Knowledge Graph** — a dated biomedical
knowledge graph built from [Open Targets](https://www.opentargets.org/) 26.03
(with Reactome, ChEMBL, and ClinicalTrials.gov), plus the code for its
**clinical-advancement benchmark**: ranking target–disease pairs by their
likelihood of advancing to Phase II, scored only from evidence datable *strictly
before* each pair's decision year.

Every temporal edge carries the year its evidence first appeared, so the graph
can be queried as of any historical decision point without leakage. Full dataset
description and statistics are in [croissant.json](croissant.json).

## Dataset

Packaged 26.03 graph + advancement benchmark are archived on Zenodo
([doi.org/10.5281/zenodo.20795232](https://doi.org/10.5281/zenodo.20795232),
CC-BY-4.0). Point the code at an unpacked copy:

```bash
export THBKG_DATA_ROOT=/path/to/opentarget_evidences
```

Paths resolve under `$THBKG_DATA_ROOT/26.03/...`; individual files can be
overridden per call (`--graph_file`, `--mappings_file`).

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv); Python ≥
3.11, PyTorch + PyG on CUDA 11.8. Training and evaluation are heavy GPU/CPU jobs,
run via SLURM (`sbatch`), not in the foreground.

```bash
uv sync
```

## Evaluate

```bash
python evaluate_advancement.py                              # all registered runs
python evaluate_advancement.py --only p3_eahgt_both,b1_hgt  # a subset
```

The primary metric is **Relative Success @ K** (an importance-weighted hit rate),
reported per therapeutic area (TA-mean over 13 areas) and Wilcoxon-tested against
a randomized-decisions baseline. RDG (Czech et al. ridge regression) and OTS
(Open Targets global score) references come from the packaged
`evaluation_dataset.zarr`. The official EA-HGT result is a grouped five-seed,
validation-selected, percentile-rank-fused ensemble.

## Rebuild from source

Four ordered SLURM stages under `scripts/` (require an Open Targets 26.03
evidence dump and the IntAct / GO / Reactome / ChEMBL / EFO sources):

```bash
sbatch scripts/collecting_edges_01.sh          # dated event lists per datasource
sbatch scripts/building_event_graph_02.sh      # HeteroData temporal + advancement edges
sbatch scripts/collecting_node_features_03.sh  # node features
sbatch scripts/assembling_graph_04.sh          # attach features → final graph
```

Then train the ensemble (per-seed configs in `config/experiments/headline/`) and
fuse:

```bash
sbatch scripts/advancement_prediction/run_grouped_ensemble_strictmask.sh
```

## Documentation

| Doc | Contents |
|-----|----------|
| [GRAPH_STRUCTURE.md](GRAPH_STRUCTURE.md) | Node/edge types, feature dims, directionality, sources. |
| [TRAINING_DETAILS.md](TRAINING_DETAILS.md) | Architectures (HGT / EA-HGT / GATv2 / LambdaRank), hyperparameters, losses, splits, metrics. |

## License

Source code: [MIT](LICENSE). Dataset artifacts: CC-BY-4.0 (see Zenodo above).

> The 23.06 build referenced in `GRAPH_STRUCTURE.md` has a same-year
> clinical-trial-edge leak and is **deprecated** — use 26.03. `GRAPH_STRUCTURE.md`
> stays useful as a type reference, but its counts and paths are for the old build.
