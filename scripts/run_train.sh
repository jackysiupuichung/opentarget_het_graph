#!/bin/bash
set -euo pipefail

source .venv/bin/activate

python -m src.train --config src/configs/test_GAT.yaml