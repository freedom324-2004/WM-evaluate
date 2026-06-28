"""
Generic API client for text-conditioned video generation.
# 用于文本控制视频生成的通用 API 客户端。
Supports async submit → poll → download pattern used by most video APIs.
# 支持大多数视频 API 使用的"异步提交 → 轮询 → 下载"模式。
"""
import base64  # 用于将图片文件转换为 base64 编码字符串，以便通过网络发送
import logging
import mimetypes
import os
import time
from typing import Dict, Any, Optional

import requests   # 非常著名的 Python 第三方 HTTP 网络请求库

logger = logging.getLogger(__name__)


class APIVideoClient:
    """
    Generic video generation API client.

    Subclass and override _submit / _poll / _parse_result for specific APIs.
    Or use directly with a compatible OpenAI-style endpoint.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        generate_path: str = "/v1/video/generate", # 新增
        status_path: str = "/v1/video/status/",    # 新增
    ):
        self.base_url = base_url or os.environ.get("VIDEO_API_URL", "")
        self.api_key = api_key or os.environ.get("VIDEO_API_KEY", "")
        self.generate_path = generate_path         # 新增保存
        self.status_path = status_path             # 新增保存
        self.timeout = timeout
        self.poll_interval = poll_interval

        self.session = requests.Session()  # 使用 requests.Session() 可以保持 HTTP 连接池，提高连续请求的性能
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

        self.headers = headers or {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode local image to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(
        self,
        model_name: str,
        prompt: str,
        image: Optional[str] = None,
        duration: float = 4.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit a generation task, poll for result, download video.
        提交生成任务，轮询等待结果，最后下载视频。
        Args:
            model_name: API model identifier
            prompt: Text prompt
            image: Path to conditioning image (for I2V)
            duration: Video duration in seconds

        Returns:
            {"code": 0, "video_path": "..."} or {"code": -1, "error": "..."}
        """
        # Submit task  第一步：提交任务（发送 Prompt 和 Image）
        task_id = self._submit(model_name, prompt, image, duration, **kwargs)
        if not task_id:
            return {"code": -1, "error": "Task submission failed"}

        # Poll for completion  第二步：死循环轮询，直到视频生成完毕拿到下载链接
        video_url = self._poll(task_id)
        if not video_url:   # 如果没拿到任务 ID，说明提交失败，直接返回错误
            return {"code": -1, "error": f"Generation failed or timed out: {task_id}"}

        # Download  第三步：下载视频
        # 在本地创建一个缓存目录，路径形如：video_cache/kling-v3/
        cache_dir = os.path.join("video_cache", model_name.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
        # 拼接本地视频文件的保存路径
        output_path = os.path.join(cache_dir, f"{task_id}.mp4")
        # 调用下载方法，如果成功则返回 code 0 和本地路径
        if self._download(video_url, output_path):
            return {"code": 0, "video_path": output_path}
        return {"code": -1, "error": f"Download failed: {video_url}"}

    def _submit(self, model_name: str, prompt: str, image: Optional[str],
                duration: float, **kwargs) -> Optional[str]:
        """Submit generation task. Override for custom APIs."""
        # 组装要发送给服务器的 JSON 数据体 (Payload)
        resolution = kwargs.pop("resolution", "720P")
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
        }
        # 如果有图片传入且文件存在，调用 encode_image 转为 base64 后加入请求体
        if image and os.path.exists(image):
            payload["image"] = self.encode_image(image)
        # 将其他可能的扩展参数一并加入
        payload.update(kwargs)

        try:
            # 发起 POST 请求，这通常是用于新建资源的 HTTP 方法
            resp = self.session.post(
                f"{self.base_url}{self.generate_path}",# 拼接提交接口的完整 URL
                headers=self.headers,  # 携带刚才组装好 Bearer 的鉴权头
                json=payload,  # 发送数据
                timeout=60,   # 60秒如果连不上就算超时
            )
            if not resp.ok:
                logger.error(f"Submit failed ({resp.status_code}): {resp.text[:500]}")
                return None
            data = resp.json()  # 将服务器返回的内容解析为 Python 字典
            task_id = data.get("task_id") or data.get("id") or data.get("data", {}).get("task_id")  # 兼容不同厂家的 API 格式，尝试从不同字段里去捞取任务 ID
            if task_id:
                logger.info(f"Task submitted: {task_id}")  # 打印成功日志
            return task_id
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return None

    def _poll(self, task_id: str) -> Optional[str]:
        """Poll task status until complete. Returns video URL."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                # 发起 GET 请求去查进度，URL 里通常会带上任务 ID 来指定查询哪个任务
                resp = self.session.get(
                    f"{self.base_url}{self.status_path}{task_id}",
                    headers=self.headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status", "")
                if status in ("succeed", "completed", "done"):
                    return data.get("video_url") or data.get("url") or data.get("result", {}).get("url")  # 尝试用多种兼容写法提取视频下载 URL，并返回
                elif status in ("failed", "error"):
                    logger.error(f"Task failed: {task_id}: {data.get('error', '')}")
                    return None

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(self.poll_interval)

        logger.error(f"Task timed out: {task_id}")
        return None

    def _download(self, url: str, output_path: str) -> bool:
        """Download video from URL with automatic retries on transient errors."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            except Exception as e:
                logger.warning(f"Download failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    time.sleep(3)
        return False


class AtlasCloudAPIClient(APIVideoClient):
    """
    Shared client for AtlasCloud platform (Kling, Seedance, Wan, Veo).
    Handles AtlasCloud-specific submit / poll logic.
    """

    def __init__(self, **kwargs):
        super().__init__(
            generate_path="/api/v1/model/generateVideo",
            status_path="/api/v1/model/prediction/",
            **kwargs,
        )

    def _submit(self, model_name: str, prompt: str, image: Optional[str],
                duration: float, **kwargs) -> Optional[str]:
        """Submit generation task to AtlasCloud."""
        duration = int(duration)  # AtlasCloud 要求 int
        resolution = kwargs.pop("resolution", "720P")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
        }

        # AtlasCloud 要求图片传入 url 或者 Base64 Data URI
        if image and os.path.exists(image):
            mime_type, _ = mimetypes.guess_type(image)
            if not mime_type:
                mime_type = "image/jpeg"
            base64_str = self.encode_image(image)
            payload["image"] = f"data:{mime_type};base64,{base64_str}"
        elif image:
            payload["image"] = image

        payload.update(kwargs)

        try:
            resp = self.session.post(
                f"{self.base_url}{self.generate_path}",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            if not resp.ok:
                logger.error(f"Submit failed ({resp.status_code}): {resp.text[:500]}")
                return None
            data = resp.json()
            # AtlasCloud 返回格式: {"data": {"id": "..."}}
            task_id = data.get("data", {}).get("id")
            if task_id:
                logger.info(f"Task submitted: {task_id}")
            else:
                logger.error(f"Submit failed: {data}")
            return task_id
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return None

    def _poll(self, task_id: str) -> Optional[str]:
        """Poll AtlasCloud task status until complete."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                resp = self.session.get(
                    f"{self.base_url}{self.status_path}{task_id}",
                    headers=self.headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                # AtlasCloud 返回格式: {"data": {"status": "...", "outputs": [...]}}
                body = data.get("data", {})
                status = body.get("status", "")

                if status in ("completed", "succeeded"):
                    outputs = body.get("outputs", [])
                    if outputs:
                        return outputs[0]  # 取第一个输出视频的 URL
                    return None

                if status == "failed":
                    logger.error(f"Task failed: {task_id}: {body.get('error', '')}")
                    return None

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(self.poll_interval)

        logger.error(f"Task timed out: {task_id}")
        return None


class KlingOfficialAPIClient(APIVideoClient):
    """Kling official API client (api-beijing.klingai.com)."""

    def __init__(self, **kwargs):
        super().__init__(
            generate_path="/v1/videos/image2video",
            status_path="/v1/videos/image2video/",
            **kwargs,
        )

    def _submit(self, model_name: str, prompt: str, image: Optional[str],
                duration: float, **kwargs) -> Optional[str]:
        duration_int = int(duration) if float(duration).is_integer() else duration
        resolution = kwargs.pop("resolution", "720P")
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": duration_int,
            "resolution": resolution,
        }
        if image and os.path.exists(image):
            payload["image"] = self.encode_image(image)
        payload.update(kwargs)

        try:
            resp = self.session.post(
                f"{self.base_url}{self.generate_path}",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            if not resp.ok:
                logger.error(f"Submit failed ({resp.status_code}): {resp.text[:500]}")
                return None
            data = resp.json()
            if data.get("code") not in (0, "0", None):
                logger.error(f"Submit failed ({data.get('code')}): {data.get('message', data)}")
                return None
            task_id = data.get("data", {}).get("task_id") or data.get("task_id")
            if task_id:
                logger.info(f"Task submitted: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return None

    def _poll(self, task_id: str) -> Optional[str]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                resp = self.session.get(
                    f"{self.base_url}{self.status_path}{task_id}",
                    headers=self.headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                body = data.get("data", data)
                status = body.get("task_status") or body.get("status", "")

                if status in ("succeed", "completed", "done"):
                    videos = body.get("task_result", {}).get("videos", [])
                    if videos:
                        return videos[0].get("url")
                    return body.get("video_url") or body.get("url")
                if status in ("failed", "error"):
                    logger.error(f"Task failed: {task_id}: {body.get('task_status_msg', '')}")
                    return None

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(self.poll_interval)

        logger.error(f"Task timed out: {task_id}")
        return None