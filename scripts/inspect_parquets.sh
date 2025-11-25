#!/usr/bin/env bash

# Root directory containing subfolders with parquet files
ROOT_DIR="$1"

if [ -z "$ROOT_DIR" ]; then
    echo "Usage: ./inspect_parquets.sh <root_dir>"
    exit 1
fi

echo "📂 Inspecting parquet files under: $ROOT_DIR"
echo

# Loop over every subdirectory
for DIR in "$ROOT_DIR"/*/; do
    # Check if directory
    [ -d "$DIR" ] || continue

    # Get first parquet file
    FILE=$(ls "$DIR"/*.parquet 2>/dev/null | head -n 1)

    if [ -z "$FILE" ]; then
        echo "⚠️  No parquet files in $DIR"
        echo
        continue
    fi

    echo "=============================================="
    echo "📁 Directory: $DIR"
    echo "📄 File: $(basename "$FILE")"
    echo "=============================================="

    # Use Python to inspect parquet
    python3 - <<EOF
import pandas as pd

file = "$FILE"

try:
    df = pd.read_parquet(file)
    print("🔹 Columns:")
    print(df.columns.tolist())
    print("\n🔸 Head:")
    print(df.head(5))
except Exception as e:
    print(f"❌ Error reading parquet: {e}")
EOF

    echo
done
