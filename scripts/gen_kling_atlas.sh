#!/bin/bash
# Kling 3.0 via AtlasCloud — generate all cases
set -e
cd "$(dirname "$0")/.."
source scripts/.env

KLING_BACKEND=atlas \
python generate.py \
    --model kling \
    --duration 4.0 \
    --resolution 720P \
    --output_dir output_videos/kling-atlas