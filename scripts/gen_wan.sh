#!/bin/bash
# Wan 2.7 — generate all cases
set -e
cd "$(dirname "$0")/.."
source scripts/.env

python generate.py \
    --model wan \
    --cases data/cases/case_40.json \
    --duration 4.0 \
    --resolution 720P \
    --output_dir output_videos/wan2.7