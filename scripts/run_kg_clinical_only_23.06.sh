#!/bin/bash
#SBATCH -J kg_clinical_only_23.06
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=64G

# Fast re-run for 23.06 after Stage B (apply_trial_resolution_23.06.py):
# re-parses nodes + chembl edges only. Leaves all other edge outputs
# untouched. Mirrors scripts/run_kg_clinical_only.sh (the 26.03 fast-path).
set -euo pipefail
source .venv/bin/activate

OT_VERSION="23.06"
INPUT="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/evidenceDated/"
OUT_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/evidences"
RAW_EDGES_DIR="${OUT_BASE}/edges"

# Drop stale chembl edge outputs so old un-bucketed `clinical_trial` files
# don't linger alongside the new bucketed files.
rm -f "${RAW_EDGES_DIR}"/*chembl*.parquet 2>/dev/null || true

python -u - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath("preprocessing/temporal_graph"))
from parsers.parser import NodeParser, EdgeParser

INPUT       = "/gpfs/scratch/bty414/opentarget_evidences/23.06/evidenceDated/"
NODE_SCHEMA = "config/node_schema.yaml"
EDGE_SCHEMA = "config/edge_schema_23.06_clinical_only.yaml"
NODE_OUT    = "/gpfs/scratch/bty414/opentarget_evidences/23.06/evidences/nodes"
EDGE_OUT    = "/gpfs/scratch/bty414/opentarget_evidences/23.06/evidences/edges"

print("🔹 Parsing nodes...", flush=True)
nodes, node_store = NodeParser(INPUT, NODE_SCHEMA, NODE_OUT, node_store=None).parse()

print("🔹 Parsing chembl edges (only)...", flush=True)
EdgeParser(INPUT, EDGE_SCHEMA, EDGE_OUT, node_store=node_store).parse()

print("✅ Done.", flush=True)
PY
