"""
Kling 3.0 video generation model (text-conditioned, I2V).

Usage:
    from src.models.text import KlingModel
    model = KlingModel()
    result = model.generate(prompt="...", image="path/to/image.jpg")
"""
import os
import logging
import time
from typing import Optional, Dict, Any

from ..base import BaseVideoModel
from .api_client import APIVideoClient
from .prompt_builder import build_turn_prompt

logger = logging.getLogger(__name__)


class KlingAPIClient(APIVideoClient):
    """Kling image-to-video API client."""

    def _submit(self, model_name: str, prompt: str, image: Optional[str],
                duration: float, **kwargs) -> Optional[str]:
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": str(int(duration)) if float(duration).is_integer() else str(duration),
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


# 继承自 BaseVideoModel。说明 KlingModel 自动拥有了基类中实现的多轮视频拼接（generate_multi_turn）等能力
class KlingModel(BaseVideoModel):
    """Kling 3.0 I2V model via API."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        super().__init__(model_name="kling")
        self._client = KlingAPIClient(
            base_url=api_url or os.environ.get("VIDEO_API_URL", "https://api-beijing.klingai.com"),
            api_key=api_key or os.environ.get("VIDEO_API_KEY", ""),
            generate_path="/v1/videos/image2video",
            status_path="/v1/videos/image2video/",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "kling-v3",
            "api_url": self._client.base_url,
            "class": "KlingModel",
        }


# 这里重写了基类的 generate 抽象方法。它是单次（单段）视频生成的真正执行者。
# 在多轮交互中，基类的 generate_multi_turn 方法会在 for 循环中反复调用这个 generate 方法。
    def generate(
        self,
        prompt: str,  # 文本提示词
        image: Optional[str] = None,  # 作为参考的输入图片路径
        **kwargs,  # 接收其他任意的关键字参数
    ) -> Dict[str, Any]:
        if not image:  # 如果没有传入图片，直接返回包含错误码和错误信息的字典，拦截请求
            return {"code": -1, "error": "Kling requires an input image (I2V only)"}
        # 校验通过，调用底层 API 客户端发起真正的网络请求
        return self._client.generate(
            model_name="kling-v3",
            prompt=prompt,
            image=image,
            duration=kwargs.get("duration", 5.0),# 从额外的 kwargs 参数中提取视频时长，如果没有专门指定，默认生成 5.0 秒的视频
        )

    def _build_turn_prompt(self, case: Dict[str, Any], interaction: Dict[str, Any],turn_index: int) -> str:
        perspective = case.get("settings", {}).get("perspective", "first_person")
        return build_turn_prompt(case, interaction, perspective=perspective,is_first_turn=(turn_index == 0))
