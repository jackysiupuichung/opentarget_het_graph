# Event-Based Temporal Graph Summary

## Overview

This document describes the event-based temporal heterogeneous graph constructed from OpenTargets data for clinical trial prediction and drug discovery research.

## Graph Structure

### Graph Type
- **Format**: PyTorch Geometric `HeteroData`
- **Paradigm**: Event-based temporal graph
- **Granularity**: `relation::datasource` level (e.g., `clinical_trial::chembl`)

### Node Types
The graph contains 5 distinct node types representing different biomedical entities:

- `disease`: Disease entities from various ontologies
- `target`: Protein/gene targets
- `molecule`: Chemical compounds and drugs
- `reactome`: Biological pathways (Reactome)
- `go`: Gene Ontology terms

**Node Dynamics**: All nodes are **static** - the node set is fixed and does not change over time. Nodes are extracted once from the complete edge set.

### Edge Types

The graph contains 27 distinct edge types at the `relation::datasource` granularity.

**Edge Dynamics**: All edges are **dynamic** (temporal) - each edge has an associated `edge_time` attribute representing when the association was first observed or when the score changed.

#### Clinical Trial Edges (Supervision Task)
- `(disease, clinical_trial::chembl, target)`: Primary supervision signal for link prediction

#### Disease-Related Edges
- `(disease, associated_with::reactome, reactome)`
- `(disease, associated_with::slapenrich, reactome)`

#### Target-Disease Association Edges
- `(target, genetic_association::eva, disease)`
- `(target, literature::europepmc, disease)`
- `(target, rna_expression::expression_atlas, disease)`
- `(target, animal_model::impc, disease)`
- `(target, genetic_association::gene_burden, disease)`
- `(target, somatic_mutation::cancer_gene_census, disease)`
- `(target, genetic_association::genomics_england, disease)`
- `(target, somatic_mutation::eva_somatic, disease)`
- `(target, genetic_association::gene2phenotype, disease)`
- `(target, genetic_association::orphanet, disease)`
- `(target, genetic_association::clingen, disease)`
- `(target, somatic_mutation::cancer_biomarkers, disease)`
- `(target, genetic_association::uniprot_variants, disease)`
- `(target, genetic_association::uniprot_literature, disease)`

#### Pathway Edges
- `(target, affected_pathway::slapenrich, disease)`
- `(target, affected_pathway::crispr_screen, disease)`
- `(target, affected_pathway::crispr, disease)`
- `(target, affected_pathway::reactome, disease)`
- `(target, affected_pathway::sysbio, disease)`
- `(target, involved_in::slapenrich, reactome)`
- `(target, involved_in::reactome, reactome)`

#### Molecular Interaction Edges
- `(target, modulated_by::chembl, molecule)`
- `(molecule, modulated_by::chembl, target)`

#### Functional Annotation Edges
- `(target, has_function_in::gene_ontology, go)`

## Temporal Properties

### Time Range
- **Start Year**: 2010
- **End Year**: 2023
- **Duration**: 14 years

### Event Compression
The pipeline compresses events by removing duplicate score updates, keeping only score-change events for each `(source, target, relation, datasource)` quadruple. This reduces storage while preserving all meaningful temporal dynamics.

## Static vs Dynamic Components

### Static Components
1. **Node Set**: Fixed set of entities (diseases, targets, molecules, pathways, GO terms)
2. **Node Types**: The 5 node types remain constant
3. **Edge Type Schema**: The 27 edge type definitions are fixed
4. **Graph Topology**: The set of possible connections is predetermined by the schema

### Dynamic Components
1. **Edge Existence**: Edges appear at specific time points (`edge_time`)
2. **Edge Weights**: Association scores (`edge_attr`) evolve over time
3. **Edge Events**: New evidence can strengthen or introduce associations
4. **Temporal Patterns**: The graph grows and evolves from 2010 to 2023

### Temporal Semantics
- **Event-Based**: Each edge represents a discrete event (e.g., "clinical trial evidence observed in year X")
- **Monotonic Growth**: Edges are never deleted, only added (knowledge accumulation)
- **Score Evolution**: Scores can change as new evidence emerges
- **Causal Ordering**: Events are strictly ordered by `edge_time`

## Edge Attributes

Each edge in the graph contains:

1. **`edge_index`**: `[2, num_edges]` tensor of source-target pairs (static structure)
2. **`edge_time`**: `[num_edges]` tensor of event timestamps in years (dynamic)
3. **`edge_attr`**: `[num_edges, 1]` tensor of edge weights/scores (dynamic)

### Score Aggregation
- **Method**: Harmonic sum (OpenTargets standard)
- **Range**: [0, 1] continuous scores
- **Interpretation**: Higher scores indicate stronger evidence of association
- **Dynamics**: Scores can increase as new evidence accumulates

## Data Splits

### Temporal Split Strategy
The graph supports temporal train/validation/test splits based on `edge_time`:

| Split | Time Range | Purpose |
|-------|------------|---------|
| **Train** | ≤ T_train | Model training on historical data |
| **Validation** | (T_train, T_val] | Hyperparameter tuning (regression metrics) |
| **Test** | > T_val | Final evaluation (ranking metrics on novel associations) |

### Snapshot Strategy
For training and inference, temporal snapshots are created by filtering edges:
- **Train Snapshot**: All edges with `edge_time ≤ T_train` (prevents future leakage)
- **Validation Snapshot**: All edges with `edge_time ≤ T_val` (for test inference context)

**Key Principle**: The model only sees edges that occurred at or before the prediction time, ensuring strict temporal causality.

## Use Cases

### Primary Task: Clinical Trial Prediction
- **Objective**: Predict novel disease-target associations for clinical trials
- **Input**: Disease node
- **Output**: Ranked list of target candidates
- **Evaluation**: Precision@k, Recall@k, NDCG@k on novel (unseen) associations
- **Temporal Aspect**: Predict future associations based on historical graph state

### Secondary Tasks
1. **Drug Repurposing**: Identify new therapeutic uses for existing molecules
2. **Target Prioritization**: Rank targets for specific diseases
3. **Pathway Analysis**: Understand disease mechanisms through pathway associations
4. **Multi-hop Reasoning**: Leverage heterogeneous graph structure for complex queries
5. **Temporal Trend Analysis**: Study how associations evolve over time

## Graph Construction Pipeline

### Step 1: Event List Generation
**Script**: `src/pipeline/build_event_list.py`

**Process**:
1. Load raw edges from parquet files
2. Filter dynamic edges by year range
3. Apply datasource-specific cutoffs
4. Aggregate scores using harmonic sum
5. Compress to score-change events only
6. Rename `year` → `edge_time`, `score` → `edge_weight`
7. Output: `output/progression/events.parquet`

### Step 2: Event Graph Building
**Script**: `src/pipeline/build_event_graph.py`

**Process**:
1. Load event list
2. Extract unique nodes from edges (static node set)
3. Build `HeteroData` object with temporal attributes
4. Assign `edge_time` and `edge_attr` to all edges
5. Output: `output/progression/temporal_graph.pt`

## Storage and Format

### File Locations
- **Event List**: `output/progression/events.parquet`
- **Graph Object**: `output/progression/temporal_graph.pt`

### Data Format
- **Event List**: Parquet table with columns: `sourceId`, `targetId`, `source_type`, `target_type`, `relation`, `datasourceId`, `edge_time`, `edge_weight`
- **Graph Object**: PyTorch Geometric `HeteroData` with temporal edge attributes

## Quality Metrics

### Data Quality
- **Completeness**: All edges have valid source/target nodes
- **Temporal Consistency**: Events are monotonically ordered within each edge type
- **Score Validity**: All scores in [0, 1] range
- **Deduplication**: No duplicate (source, target, relation, datasource, time) tuples

### Temporal Properties
- **Causality**: Strict temporal ordering ensures no future leakage
- **Granularity**: Year-level temporal resolution
- **Coverage**: Spans multiple years of biomedical knowledge evolution
- **Density**: Graph becomes denser over time as knowledge accumulates

## Comparison to Snapshot Approach

| Aspect | Event-Based (Current) | Snapshot-Based (Legacy) |
|--------|----------------------|-------------------------|
| **Storage** | Single graph file | Multiple yearly files |
| **Flexibility** | Any temporal split | Fixed yearly splits |
| **Granularity** | Event-level | Year-level |
| **Causality** | Exact event ordering | Approximate (yearly) |
| **Efficiency** | High (single load) | Lower (multiple loads) |
| **Node Set** | Static (extracted once) | Static per snapshot |
| **Edge Set** | Dynamic (time-stamped) | Static per snapshot |

## Training Strategy

### Snapshot-Based Temporal Sampling
Instead of using complex temporal samplers that require specialized dependencies, we use a **snapshot filtering strategy**:

1. **Pre-filter Graph**: Create `train_snapshot = filter_graph_by_time(graph, T_train)`
2. **Static Sampling**: Use standard `LinkNeighborLoader` on the filtered snapshot
3. **No Leakage**: Since all edges in `train_snapshot` have `edge_time ≤ T_train`, standard sampling is temporally safe
4. **Efficiency**: Avoids dependency on `pyg-lib` temporal samplers while maintaining correctness

### Evaluation Strategy
- **Validation**: Regression metrics (MSE) on score prediction
- **Test**: Ranking metrics (P@k, R@k, NDCG@k) on novel association discovery
- **Context**: Use appropriate temporal snapshot for embedding generation (strict no-leakage)

## Future Extensions

### Potential Enhancements
1. **Finer Temporal Granularity**: Month or day-level timestamps
2. **Edge Features**: Add publication metadata, trial phase information
3. **Dynamic Node Features**: Incorporate time-varying node attributes (e.g., gene expression over time)
4. **Temporal Attention**: Weight edges by recency or importance
5. **Multi-relational Paths**: Explicit path-based features for reasoning
6. **Node Dynamics**: Allow nodes to appear/disappear over time (currently static)

### Scalability Considerations
- Current graph fits comfortably in memory
- Snapshot filtering is efficient for moderate-scale graphs
- For larger scales, consider:
  - Graph sampling strategies
  - Distributed training
  - Approximate nearest neighbor search
  - Incremental learning approaches

## References

- **OpenTargets Platform**: https://platform.opentargets.org/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **HGT Paper**: "Heterogeneous Graph Transformer" (WWW 2020)
- **Temporal Graphs**: "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (ICML 2020)

---

**Last Updated**: 2026-01-22  
**Graph Version**: v1.0  
**Pipeline**: Event-based with snapshot filtering
