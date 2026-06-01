#!/bin/bash
#SBATCH -J kg_clinical_only
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -t 6:0:0
#SBATCH --mem-per-cpu=64G

# Fast re-run after Stage B (apply_trial_resolution.py):
# re-parses nodes + clinical_precedence edges only, leaves all other edge
# outputs untouched.
set -euo pipefail
source .venv/bin/activate

OT_VERSION="${OT_VERSION:-26.03}"
INPUT="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/evidenceDated/"
OUT_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/evidences"
RAW_EDGES_DIR="${OUT_BASE}/edges"
RAW_NODES_DIR="${OUT_BASE}/nodes"

# Drop stale clinical edge outputs so old `clinical_trial` files don't linger
# alongside the new bucketed files.
rm -f "${RAW_EDGES_DIR}"/*clinical_precedence*.parquet 2>/dev/null || true

python -u - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath("preprocessing/temporal_graph"))
from parsers.parser import NodeParser, EdgeParser

INPUT = "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidenceDated/"
NODE_SCHEMA  = "config/node_schema.yaml"
EDGE_SCHEMA  = "config/edge_schema_26.03_clinical_only.yaml"
NODE_OUT     = "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidences/nodes"
EDGE_OUT     = "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidences/edges"

print("🔹 Parsing nodes...", flush=True)
nodes, node_store = NodeParser(INPUT, NODE_SCHEMA, NODE_OUT, node_store=None).parse()

print("🔹 Parsing clinical_precedence edges (only)...", flush=True)
EdgeParser(INPUT, EDGE_SCHEMA, EDGE_OUT, node_store=node_store).parse()

print("✅ Done.", flush=True)
PY
