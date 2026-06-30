"""
Kling 3.0 video generation model (text-conditioned, I2V).

Supports two backends:
- KLING_BACKEND=official  → uses KlingOfficialAPIClient (api-beijing.klingai.com)
- KLING_BACKEND=atlas    → uses AtlasCloudAPIClient       (atlascloud.ai)

Usage:
    from src.models.text import KlingModel
    model = KlingModel()
    result = model.generate(prompt="...", image="path/to/image.jpg")
"""
import logging
import os
from typing import Dict, Any, Optional

from ..base import BaseVideoModel
from .api_client import AtlasCloudAPIClient, KlingOfficialAPIClient
from .prompt_builder import build_turn_prompt

logger = logging.getLogger(__name__)


class KlingModel(BaseVideoModel):
    """Kling 3.0 I2V model via Kling official API or AtlasCloud."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        super().__init__(model_name="kling")
        backend = os.environ.get("KLING_BACKEND", "official")

        if backend == "atlas":
            logger.info("KlingModel: using AtlasCloud backend")
            self._client = AtlasCloudAPIClient(
                base_url=api_url or os.environ.get("VIDEO_API_URL", "https://api.atlascloud.ai"),
                api_key=api_key or os.environ.get("VIDEO_API_KEY", ""),
            )
        else:
            logger.info("KlingModel: using Kling official backend")
            self._client = KlingOfficialAPIClient(
                base_url=api_url or os.environ.get("VIDEO_API_URL", "https://api-beijing.klingai.com"),
                api_key=api_key or os.environ.get("VIDEO_API_KEY", ""),
            )

    def get_model_info(self) -> Dict[str, Any]:
        backend = os.environ.get("KLING_BACKEND", "official")
        model_name = "kwaivgi/kling-v3.0-std/image-to-video" if backend == "atlas" else "kling-v3"
        return {
            "model_name": model_name,
            "api_url": self._client.base_url,
            "class": "KlingModel",
            "backend": backend,
        }

    def generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not image:
            return {"code": -1, "error": "Kling requires an input image (I2V only)"}
        backend = os.environ.get("KLING_BACKEND", "official")
        model_name = "kwaivgi/kling-v3.0-std/image-to-video" if backend == "atlas" else "kling-v3"
        output_path = kwargs.pop("output_path", None)
        meta_keys = {"turn", "action", "interaction_type"}
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}
        return self._client.generate(
            model_name=model_name,
            prompt=prompt,
            image=image,
            output_path=output_path,
            duration=gen_kwargs.get("duration", 4.0),
            resolution=gen_kwargs.get("resolution", "720P"),
        )

    def _build_turn_prompt(self, case: Dict[str, Any], interaction: Dict[str, Any], turn_index: int) -> str:
        perspective = case.get("settings", {}).get("perspective", "first_person")
        return build_turn_prompt(case, interaction, perspective=perspective, is_first_turn=(turn_index == 0))