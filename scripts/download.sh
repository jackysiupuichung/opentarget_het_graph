#!/bin/bash
BASE=ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.06/output/etl/parquet/evidence
OUTDIR=./evidence_first_files

mkdir -p "$OUTDIR"

# Get list of subdirectories
for subdir in $(wget -qO- $BASE/ | grep -o 'sourceId=[^/"]*'); do
  echo "Processing $subdir ..."

  # Get first parquet file inside subdir
  first_file=$(wget -qO- $BASE/$subdir/ \
                  | grep -o 'part-[^"]*\.parquet' \
                  | sort | head -n1)

  if [ -n "$first_file" ]; then
    echo "Downloading $first_file from $subdir"
    mkdir -p "$OUTDIR/$subdir"
    wget --no-parent --no-host-directories --cut-dirs=7 \
         -P "$OUTDIR/$subdir" \
         "$BASE/$subdir/$first_file"
  else
    echo "⚠️ No parquet files found in $subdir"
  fi
done

