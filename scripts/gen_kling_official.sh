#!/bin/bash
# Kling 3.0 via Kling Official API — generate all cases
#
# Unlike AtlasCloud which uses a static API key, the Kling official API
# requires a JWT token generated from access_key + secret_key (valid 24h).
# This script generates the JWT on-the-fly before calling generate.py.
# 与 AtlasCloud 使用固定 API Key 不同，Kling 官方 API 需要从
# access_key + secret_key 生成 JWT token（有效期 24 小时）。
# 本脚本在执行 generate.py 前自动生成 JWT。

set -e
cd "$(dirname "$0")/.."
source scripts/.env

# 自动从 KLING_ACCESS_KEY + KLING_SECRET_KEY 生成 JWT token
export VIDEO_API_KEY=$(python3 -c "
import time, jwt, os

ak = os.environ['KLING_ACCESS_KEY']
sk = os.environ['KLING_SECRET_KEY']

headers = {'alg': 'HS256', 'typ': 'JWT'}
payload = {
    'iss': ak,
    'exp': int(time.time()) + 86400,   # 24 小时后过期
    'nbf': int(time.time()) - 5,       # 立即生效（允许 5 秒偏差）
}
token = jwt.encode(payload, sk, headers=headers)
print(token)
")

VIDEO_API_URL="https://api-beijing.klingai.com" \
python generate.py \
    --model kling \
    --duration 3.0 \
    --resolution 720P \
    --output_dir output_videos/kling-official