# Data Parser Summary

## Overview

The OpenTargets heterogeneous graph is constructed from multiple biomedical data sources using a modular parsing system. Each parser extracts specific types of associations from OpenTargets Platform data.

## Parser Architecture

### Base Parser (`BaseParser`)
- **Location**: `src/parsers/parser.py`
- **Function**: Abstract base class providing common functionality
- **Features**:
  - Schema-driven parsing from YAML configuration
  - Parquet file I/O
  - Property normalization and validation

### Node Parser (`NodeParser`)
- **Location**: `src/parsers/parser.py`
- **Function**: Extracts node entities (diseases, targets, molecules, pathways, GO terms)
- **Output**: Node tables with unique IDs
- **Special Handling**: Filters targets to protein-coding genes only

### Edge Parser (`EdgeParser`)
- **Location**: `src/parsers/parser.py`
- **Function**: Extracts relationships between entities
- **Features**:
  - Handles nested data structures (lists, dicts)
  - Extracts temporal information (publication years, study dates)
  - Fetches PubMed publication years via NCBI E-utilities API
  - Validates edges against node store
  - Supports both static and dynamic (temporal) edges

## Data Sources Parsed

### 1. Clinical Trial Evidence
**Parser**: `chembl`  
**Source**: ChEMBL clinical trials database  
**Edges Generated**:
- `(disease, clinical_trial, target)` - Disease-target associations from clinical trials
- `(target, modulated_by, molecule)` - Target-drug modulation relationships

**Temporal Information**: Study start dates, publication years  
**Properties**: Clinical status, study stop reasons, trial scores

---

### 2. Genetic Associations

#### EVA (European Variation Archive)
**Parser**: `eva`  
**Source**: Common and rare genetic variants  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Publication year from literature  
**Evidence**: Variant-disease associations from GWAS and other studies

#### Gene Burden
**Parser**: `gene_burden`  
**Source**: Rare variant burden analysis  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Study/publication year  
**Evidence**: Statistical evidence from rare variant aggregation tests

#### ClinGen
**Parser**: `clingen`  
**Source**: Clinical Genome Resource  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Curation year  
**Evidence**: Expert-curated gene-disease validity classifications

#### Genomics England PanelApp
**Parser**: `genomics_england`  
**Source**: Gene panels for rare diseases  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Panel version/curation date  
**Evidence**: Expert-reviewed gene-disease panels

#### Orphanet
**Parser**: `orphanet`  
**Source**: Rare disease database  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Database version  
**Evidence**: Gene-disease associations for rare diseases

#### Gene2Phenotype
**Parser**: `gene2phenotype`  
**Source**: Developmental disorders database  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Curation date  
**Evidence**: Curated gene-phenotype relationships

#### UniProt Literature
**Parser**: `uniprot_literature`  
**Source**: UniProt protein database  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: PubMed publication year  
**Evidence**: Literature-derived gene-disease associations

#### UniProt Variants
**Parser**: `uniprot_variants`  
**Source**: UniProt variant annotations  
**Edge**: `(target, genetic_association, disease)`  
**Temporal**: Annotation date  
**Evidence**: Pathogenic variant annotations

---

### 3. Somatic Mutations (Cancer)

#### Cancer Gene Census
**Parser**: `cancer_gene_census`  
**Source**: COSMIC Cancer Gene Census  
**Edge**: `(target, somatic_mutation, disease)`  
**Temporal**: Publication year  
**Evidence**: Cancer driver genes and mutations

#### EVA Somatic
**Parser**: `eva_somatic`  
**Source**: Somatic variant database  
**Edge**: `(target, somatic_mutation, disease)`  
**Temporal**: Study publication year  
**Evidence**: Somatic mutations in cancer

#### Cancer Biomarkers
**Parser**: `cancer_biomarkers`  
**Source**: Cancer biomarker database  
**Edge**: `(target, somatic_mutation, disease)`  
**Temporal**: Publication/approval year  
**Evidence**: Validated cancer biomarkers

---

### 4. Pathway and Functional Associations

#### Reactome
**Parser**: `reactome`  
**Source**: Reactome pathway database  
**Edges Generated**:
- `(target, affected_pathway, disease)` - Pathway dysregulation in disease
- `(target, involved_in, reactome)` - Gene-pathway membership
- `(disease, associated_with, reactome)` - Disease-pathway associations

**Temporal**: Pathway version/curation date  
**Evidence**: Curated biological pathways

#### SLAPenrich
**Parser**: `slapenrich`  
**Source**: Pathway enrichment analysis  
**Edges Generated**:
- `(target, affected_pathway, disease)` - Enriched pathways
- `(target, involved_in, reactome)` - Gene-pathway membership
- `(disease, associated_with, reactome)` - Disease-pathway enrichment

**Temporal**: Analysis date  
**Evidence**: Statistical pathway enrichment

#### CRISPR Screens
**Parser**: `crispr`  
**Source**: CRISPR knockout screens  
**Edge**: `(target, affected_pathway, disease)`  
**Temporal**: Publication year  
**Evidence**: Essential genes from CRISPR screens

**Parser**: `crispr_screen`  
**Source**: Large-scale CRISPR screens  
**Edge**: `(target, affected_pathway, disease)`  
**Temporal**: Publication year  
**Evidence**: Systematic CRISPR screening data

#### SysBio
**Parser**: `sysbio`  
**Source**: Systems biology analyses  
**Edge**: `(target, affected_pathway, disease)`  
**Temporal**: Study publication year  
**Evidence**: Network-based disease mechanisms

---

### 5. Expression and Model Organisms

#### Expression Atlas
**Parser**: `expression_atlas`  
**Source**: Gene expression database  
**Edge**: `(target, rna_expression, disease)`  
**Temporal**: Study publication year  
**Evidence**: Differential gene expression in disease

#### IMPC (International Mouse Phenotyping Consortium)
**Parser**: `impc`  
**Source**: Mouse knockout phenotypes  
**Edge**: `(target, animal_model, disease)`  
**Temporal**: Phenotyping date  
**Evidence**: Mouse model phenotypes matching human diseases

---

### 6. Literature Mining

#### Europe PMC
**Parser**: `europepmc`  
**Source**: Text-mined literature  
**Edge**: `(target, literature, disease)`  
**Temporal**: PubMed publication year (fetched via API)  
**Evidence**: Co-occurrence of genes and diseases in scientific literature

---

### 7. Functional Annotations

#### Gene Ontology
**Parser**: `gene_ontology`  
**Source**: GO annotations from targets file  
**Edge**: `(target, has_function_in, go)`  
**Temporal**: Annotation date  
**Evidence**: Functional annotations of genes

---

## Temporal Information Handling

### Year Extraction Priority
The parser attempts to extract temporal information in the following order:
1. **`curationYear`**: Expert curation date (highest priority)
2. **`studyYear`**: Study/analysis year
3. **`publicationYear`**: Publication year
4. **`studyStartDate`**: Clinical trial start date (converted to year)

### PubMed API Integration
For edges with literature references but no explicit year:
- Extracts PubMed IDs (PMIDs) from nested structures
- Batches API requests to NCBI E-utilities (10,000 PMIDs per batch)
- Caches results to avoid redundant API calls
- Uses oldest publication year when multiple references exist

### Static vs Dynamic Edges
- **Dynamic Edges**: Have temporal information (year attribute)
- **Static Edges**: No temporal dimension (e.g., some GO annotations)

## Data Quality and Validation

### Node Validation
- Ensures all edge endpoints exist in the node store
- Filters out edges with invalid source or target IDs
- Protein-coding gene filter for targets

### Edge Validation
- Removes edges missing required columns
- Validates score ranges [0, 1]
- Deduplicates based on (source, target, relation, datasource)
- Handles nested data structures (lists, dicts)

### Score Handling
- Some datasources provide scores (e.g., ChEMBL, genetic associations)
- Others use constant scores (e.g., GO annotations = 1.0)
- Scores represent evidence strength or statistical significance

## Output Format

### Edge Schema
All parsers produce edges with the following columns:
- `sourceId`: Source entity ID
- `targetId`: Target entity ID
- `source_type`: Source node type (disease, target, molecule, etc.)
- `target_type`: Target node type
- `relation`: Relationship type (clinical_trial, genetic_association, etc.)
- `datasourceId`: Data source identifier
- `score`: Evidence score [0, 1]
- `year`: Temporal information (for dynamic edges)
- Additional properties (e.g., `clinicalStatus`, `literature`)

### File Naming Convention
Output files follow the pattern:
```
{source_type}_{relation}_{target_type}_{datasource}.parquet
```

Example: `disease_clinical_trial_target_chembl.parquet`

## Parser Configuration

### Schema Files
- **Edge Schema**: `config/edge_schema.yaml`
- **Node Schema**: `config/node_schema.yaml`

### Customization
Each parser entry in the schema specifies:
- `input_dir`: Source data directory
- `sourceId`: Column for source entity
- `targetId`: Column for target entity
- `relation_name`: Relationship type
- `props`: Additional properties to extract

## Summary Statistics

| Category | Number of Parsers | Edge Types |
|----------|------------------|------------|
| Clinical Trials | 1 | 2 |
| Genetic Associations | 8 | 8 |
| Somatic Mutations | 3 | 3 |
| Pathways | 5 | 9 |
| Expression/Models | 2 | 2 |
| Literature | 1 | 1 |
| Functional | 1 | 1 |
| **Total** | **21** | **26** |

Note: Some parsers generate multiple edge types (e.g., Reactome generates 3 different edge types).

## Special Parsers

### ChEMBL Trial Expander
**Location**: `src/parsers/chembl_trial_expander.py`  
**Function**: Expands clinical trial data with additional metadata  
**Features**: Enriches trial information with phase, status, and outcome data

### IntAct Parser
**Location**: `src/parsers/intact/`  
**Function**: Parses protein-protein interaction data  
**Status**: Supplementary parser (not in main pipeline)

### GO Ontology Parser
**Location**: `src/parsers/go_ontology/`  
**Function**: Processes Gene Ontology hierarchy  
**Status**: Supplementary parser for ontology structure

---

**Last Updated**: 2026-01-22  
**Schema Version**: v1.0  
**Total Data Sources**: 21 unique sources
