#!/bin/bash
# Veo 3.1 Fast — generate all cases
set -e
cd "$(dirname "$0")/.."
source scripts/.env

python generate.py \
    --model veo \
    --cases data/cases/case_40.json \
    --duration 4.0 \
    --resolution 720P \
    --output_dir output_videos/veo3.1-fast