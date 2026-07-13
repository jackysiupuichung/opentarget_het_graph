# opentarget-het-graph

A heterogeneous temporal knowledge graph built from [Open Targets](https://www.opentargets.org/)
evidence, together with the models, training, and evaluation code for the
**drug-advancement** task: predicting which target–disease pairs will advance to a
higher clinical phase, evaluated under decision-aligned temporal masking.

The graph combines genetic, molecular, literature, pathway, and clinical-trial
evidence across time (1995–2025) into a single directed `HeteroData` object where
every temporal edge carries a snapshot year and a 2-dim evidence score. Models are
trained to rank advancement candidates using only evidence available before each
pair's decision year.

## License

The **source code** in this repository is released under the [MIT License](LICENSE).
The **THBKG dataset artifacts** (the Open Targets 26.03 graph and EA-HGT
advancement benchmark, archived at
[doi.org/10.5281/zenodo.20795232](https://doi.org/10.5281/zenodo.20795232))
are licensed separately under Creative Commons Attribution 4.0 International
(CC-BY-4.0).

## Graph release

The **canonical graph is the Open Targets 26.03 build** (20 relation types,
schema in [config/edge_schema_26.03.yaml](config/edge_schema_26.03.yaml)).

> **Note:** The 23.06 build described in [GRAPH_STRUCTURE.md](GRAPH_STRUCTURE.md)
> contains a same-year clinical-trial-edge leak and is **deprecated** — all 23.06
> results are invalid. Use 26.03. `GRAPH_STRUCTURE.md` remains useful as a
> node/edge-type reference but its counts and file path are for the old release.

## Documentation

| Doc | Contents |
|-----|----------|
| [GRAPH_STRUCTURE.md](GRAPH_STRUCTURE.md) | Node types, edge (relation) types, feature dims, directionality, data sources. (23.06 reference — see note above.) |
| [TRAINING_DETAILS.md](TRAINING_DETAILS.md) | Model architectures (HGT / EA-HGT / GATv2 / LambdaRank), hyperparameters, losses, splits, metrics, Optuna search spaces. |

## Repository layout

```
config/           Graph schemas (node/edge/static) and experiment configs
  edge_schema_26.03.yaml     Canonical relation schema
  experiments/               Per-experiment training configs (sweeps, seeds, ...)
preprocessing/    Temporal-graph construction from Open Targets evidence
data/             Derived tables (clinical-trial advancement, NCT dates, validation diseases)
src/
  data/           temporal_loader.py — event-graph loading + temporal neighbor sampling
  models/         HGT (+ RTE / edge-aware), GATv2/v3, RGCN, CompGCN, decoder, time encoder
  losses/         Focal BCE, LambdaRank
  eval/           Prospective / relative-success evaluation
  explain/        Post-hoc explanation: integrated gradients, attention, PaGE-Link
  train_advancement_hgt.py         Focal-BCE advancement training
  train_advancement_lambdarank.py  LambdaRank (ranking) advancement training
  train_target_discovery.py        Target-discovery task
evaluate_advancement.py            Main advancement evaluation entrypoint
```

## Task and evaluation

- **Task:** link prediction on the `advancement` edge (target → disease), framed as
  ranking candidate pairs by predicted advancement.
- **Temporal masking:** for each query pair, only evidence dated strictly before the
  pair's decision year is visible during sampling — no same-year or future leakage.
- **Primary metric:** Relative Success @ K (RS@K), an importance-weighted hit rate,
  reported per therapeutic area and Wilcoxon-tested against a randomized-decisions
  baseline. Secondary: NDCG@K, precision/recall@K, and classification metrics.

The official EA-HGT result is a grouped 5-seed rank-averaged ensemble; see
[TRAINING_DETAILS.md](TRAINING_DETAILS.md) for per-model hyperparameters.

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv) (see
[pyproject.toml](pyproject.toml) and `uv.lock`); Python ≥ 3.11, PyTorch + PyG on
CUDA 11.8.

```bash
uv sync
```

Training and evaluation are heavy GPU/CPU jobs and are run via SLURM (`sbatch`),
not in the foreground.

### Data location

Large artifacts (the built graph, node mappings, and per-seed run
checkpoints/predictions) live **outside** the repo. Two ways to obtain them:

1. **Download the release** — the packaged 26.03 graph + advancement benchmark
   is archived on Zenodo ([doi.org/10.5281/zenodo.20795232](https://doi.org/10.5281/zenodo.20795232),
   CC-BY-4.0). Unpack it and point the code at it with:

   ```bash
   export THBKG_DATA_ROOT=/path/to/opentarget_evidences
   ```

   `evaluate_advancement.py` resolves its graph, mappings, and run registry
   under `$THBKG_DATA_ROOT/26.03/...` (falling back to the author's cluster
   scratch if unset). Individual paths can also be overridden per call, e.g.
   `--graph_file` / `--mappings_file`.

2. **Rebuild from source** — see the pipeline below.

## Reproduce

### 1. Build the graph from Open Targets evidence

The graph is built in four stages (each a SLURM script under `scripts/`,
run in order). They require an Open Targets 26.03 evidence dump and the
IntAct / GO / Reactome / ChEMBL / EFO sources described in
[GRAPH_STRUCTURE.md](GRAPH_STRUCTURE.md):

```bash
sbatch scripts/collecting_edges_01.sh        # dated event lists per datasource
sbatch scripts/building_event_graph_02.sh    # HeteroData temporal + advancement edges
sbatch scripts/collecting_node_features_03.sh # target/disease/molecule/GO/Reactome features
sbatch scripts/assembling_graph_04.sh         # attach features -> final graph
```

Temporal scoring (harmonic-sum cumulative score + novelty) and the strict
`<` as-of masking are applied during stages 1–2; constants live in
`preprocessing/.../build_event_list.py`.

### 2. Train the ensemble

The headline EA-HGT result is a grouped five-seed (1, 7, 42, 123, 2024),
validation-selected, percentile-rank-fused ensemble. Per-seed configs are in
`config/experiments/headline/`; launch the per-seed run scripts under
`scripts/advancement_prediction/headline/`, then fuse:

```bash
sbatch scripts/advancement_prediction/run_grouped_ensemble_strictmask.sh
```

Set `SAVE_PER_EPOCH_PREDS=1` during training so the ensemble builder can
select each seed's checkpoint by validation per-area median RS@50.

### 3. Evaluate

```bash
python evaluate_advancement.py                 # all registered runs with predictions
python evaluate_advancement.py --only p3_eahgt_both,b1_hgt   # a subset
```

RS@K, NDCG@K, and classification metrics are reported per therapeutic area
(TA-mean over the 13 retained areas) with Wilcoxon tests against a
randomized-decisions baseline. RDG (Czech et al. ridge regression) and OTS
(Open Targets global score) references are read from the packaged
`evaluation_dataset.zarr`.
