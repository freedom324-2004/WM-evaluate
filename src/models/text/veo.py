"""
Veo 3.1 Fast video generation model (I2V via AtlasCloud).

Usage:
    from src.models.text import VeoModel
    model = VeoModel()
    result = model.generate(prompt="...", image="path/to/image.jpg")
"""
import os
from typing import Optional, Dict, Any

from ..base import BaseVideoModel
from .api_client import AtlasCloudAPIClient
from .prompt_builder import build_turn_prompt


class VeoModel(BaseVideoModel):
    """Veo 3.1 Fast I2V model via AtlasCloud."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        super().__init__(model_name="veo")
        self._client = AtlasCloudAPIClient(
            base_url=api_url or os.environ.get("VIDEO_API_URL", "https://api.atlascloud.ai"),
            api_key=api_key or os.environ.get("VIDEO_API_KEY", ""),
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "veo3.1-fast",
            "api_url": self._client.base_url,
            "class": "VeoModel",
        }

    def generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        output_path = kwargs.pop("output_path", None)
        meta_keys = {"turn", "action", "interaction_type"}
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}
        return self._client.generate(
            model_name="google/veo3.1-fast/image-to-video",
            prompt=prompt,
            image=image,
            output_path=output_path,
            duration=int(gen_kwargs.get("duration", 4)),
            resolution=gen_kwargs.get("resolution", "720p").lower(),
            aspect_ratio=gen_kwargs.get("aspect_ratio", "16:9"),
            generate_audio=gen_kwargs.get("generate_audio", False),
            seed=gen_kwargs.get("seed", -1),
            negative_prompt=gen_kwargs.get("negative_prompt", ""),
        )

    def _build_turn_prompt(self, case: Dict[str, Any], interaction: Dict[str, Any], turn_index: int) -> str:
        perspective = case.get("settings", {}).get("perspective", "first_person")
        return build_turn_prompt(case, interaction, perspective=perspective, is_first_turn=(turn_index == 0))
