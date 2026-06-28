# lxw
# Access Key: AK3Edn834ffAkNHhLDBQRyFNKEYkme3J
# Secret Key: DrA8yPBnk8nkry4fEL3e8Q4fyFreAbJ9



import time
import jwt  # PyJWT 库，用于生成 JSON Web Token (JWT)，需要提前安装（pip install PyJWT）

ak = "AK3Edn834ffAkNHhLDBQRyFNKEYkme3J" # 填写access key
sk = "DrA8yPBnk8nkry4fEL3e8Q4fyFreAbJ9" # 填写secret key

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 86400, # 有效时间，此处示例代表当前时间+86400s(24小时)
        "nbf": int(time.time()) - 5 # 开始生效的时间，此处示例代表当前时间-5秒
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

api_token = encode_jwt_token(ak, sk)
print(api_token) # 打印生成的API_TOKEN




""" 
KLING_BACKEND=official \
VIDEO_API_KEY="api_key_here" \
VIDEO_API_URL="https://api-beijing.klingai.com" \
python generate.py \
    --model kling \
    --cases data/cases/case_40.json \
    --duration 3.0 \
    --output_dir output_videos/kling-official 
"""
  
  
  
    
""" 
KLING_BACKEND=atlas \
VIDEO_API_KEY="api_key_here" \
VIDEO_API_URL="https://api.atlascloud.ai" \
python generate.py \
    --model kling \
    --cases data/cases/case_40.json \
    --duration 3.0 \
    --output_dir output_videos/kling-atlas
 """    