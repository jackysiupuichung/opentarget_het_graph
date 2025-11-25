#!/bin/bash

BASE="http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.06/output/etl/parquet/"
OUTDIR="/data/scratch/bty414/opentarget_evidences/23.06/evidenceDated"

# What to download
# 25.06
# NODES=(
#     "disease"
#     "target"
#     "drug_molecule"
#     "reactome"
#     "go"
#     "interaction_evidence"
# )

# 23.06
NODES=(
    "diseases"
    "targets"
    # "molecule"
    # "reactome"
    # "go"
    # "interactionEvidence"
)

# # Rename mapping
# declare -A RENAME_MAP=(
#     ["target"]="targets"
#     ["disease"]="diseases"
# )

mkdir -p "$OUTDIR"

echo "📡 Starting OpenTargets 25.06 node download..."

for node in "${NODES[@]}"; do

    # Apply rename if exists
    if [[ -n "${RENAME_MAP[$node]}" ]]; then
        OUT_NAME="${RENAME_MAP[$node]}"
    else
        OUT_NAME="$node"
    fi

    NODE_URL="$BASE/$node"
    NODE_OUT="$OUTDIR/$OUT_NAME"

    mkdir -p "$NODE_OUT"

    echo ""
    echo "============================"
    echo "📥 Downloading node: $node  → saved as: $OUT_NAME"
    echo "URL: $NODE_URL"
    echo "============================"

    parquet_files=$(wget -qO- "$NODE_URL/" | grep -o '[^"]*\.parquet')

    if [ -z "$parquet_files" ]; then
        echo "⚠️ No parquet files found for $node"
        continue
    fi

    for file in $parquet_files; do
        echo "⬇️ $file"
        wget -nc -P "$NODE_OUT" "$NODE_URL/$file"
    done

    echo "✔️ Done $node → $OUT_NAME"
done

echo ""
echo "🎉 Completed node downloads → $OUTDIR"
