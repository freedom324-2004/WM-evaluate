"""
Seedance 2.0 video generation model (I2V via AtlasCloud).

Usage:
    from src.models.text import SeedanceModel
    model = SeedanceModel()
    result = model.generate(prompt="...", image="path/to/image.jpg")
"""
import os
from typing import Optional, Dict, Any

from ..base import BaseVideoModel
from .api_client import AtlasCloudAPIClient
from .prompt_builder import build_turn_prompt


class SeedanceModel(BaseVideoModel):
    """Seedance 2.0 I2V model via AtlasCloud."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        super().__init__(model_name="seedance")
        self._client = AtlasCloudAPIClient(
            base_url=api_url or os.environ.get("VIDEO_API_URL", "https://api.atlascloud.ai"),
            api_key=api_key or os.environ.get("VIDEO_API_KEY", ""),
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "seedance-2.0",
            "api_url": self._client.base_url,
            "class": "SeedanceModel",
        }

    def generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_name = "bytedance/seedance-2.0/image-to-video"
        # Extract output_path and strip multi-turn bookkeeping kwargs
        output_path = kwargs.pop("output_path", None)
        meta_keys = {"turn", "action", "interaction_type"}
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}
        return self._client.generate(
            model_name=model_name,
            prompt=prompt,
            image=image,
            output_path=output_path,
            duration=int(gen_kwargs.get("duration", 4)),
            resolution=gen_kwargs.get("resolution", "720p").lower(),
            generate_audio=False,
        )

    def _build_turn_prompt(self, case: Dict[str, Any], interaction: Dict[str, Any], turn_index: int) -> str:
        perspective = case.get("settings", {}).get("perspective", "first_person")
        return build_turn_prompt(case, interaction, perspective=perspective, is_first_turn=(turn_index == 0))
