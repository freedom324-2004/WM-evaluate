"""
CogVideoX1.5-5B-I2V local implementation.
CogVideoX1.5-5B-I2V 本地实现。

Uses the diffusers CogVideoXImageToVideoPipeline to run inference
with the THUDM/CogVideoX1.5-5B-I2V weights.
使用 diffusers 的 CogVideoXImageToVideoPipeline 配合
THUDM/CogVideoX1.5-5B-I2V 权重进行推理。

Model card: https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V
模型卡片：https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V
"""
from typing import Optional, List, Dict, Any
import os

import numpy as np
import torch
from PIL import Image

from .base import LocalVideoModel
from ..text.prompt_builder import build_turn_prompt


# ---------------------------------------------------------------------------
# Configuration  配置
# ---------------------------------------------------------------------------

# Default model identifier — can be a HuggingFace repo ID or a local path.
# 默认模型标识符 — 可以是 HuggingFace 仓库 ID 或本地路径。
# Change this if you want to use a different checkpoint (e.g. kijai variant).
# 如需使用不同的检查点（如 kijai 变体），请修改此项。
DEFAULT_MODEL_ID = "/root/autodl-tmp/models/CogVideoX1.5-5B-I2V"

# Number of frames generated per call.
# 每次调用生成的帧数。
#   - CogVideoX1.5-5B-I2V default produces 49 frames (~2 s at 24 fps).
#     CogVideoX1.5-5B-I2V 默认生成 49 帧（约 2 秒 @ 24 fps）。
#   - Setting lower values may reduce quality.
#     设置较低的值可能会降低质量。
DEFAULT_NUM_FRAMES = 49

# ---------------------------------------------------------------------------
# Model  模型
# ---------------------------------------------------------------------------


class CogVideoModel(LocalVideoModel):
    """CogVideoX1.5-5B-I2V running locally via diffusers.
    CogVideoX1.5-5B-I2V 通过 diffusers 在本地运行。"""

    def __init__(
        self,
        model_name: str = "cogvideo",
        device: str = "cuda",
        model_id: str = DEFAULT_MODEL_ID,
        torch_dtype: torch.dtype = torch.bfloat16,
        variant: Optional[str] = None,
        # ---- Memory management ----  内存管理
        enable_cpu_offload: bool = True,
    ):
        super().__init__(model_name, device=device)
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.variant = variant
        self._pipe = None   # will hold the pipeline after _load_model  加载模型后将持有 pipeline

        # Seed generator for reproducibility
        # 用于可复现性的种子生成器
        self._generator = None

        # ---- Offload control ----  卸载控制
        self.enable_cpu_offload = enable_cpu_offload

    # -- Model lifecycle --------------------------------------------------
    # -- 模型生命周期 --------------------------------------------------

    def _load_model(self):
        """Load the CogVideoX I2V pipeline (once).
        加载 CogVideoX I2V 管线（仅一次）。"""
        from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
            CogVideoXImageToVideoPipeline,
        )

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            variant=self.variant,
        )

        # -- Memory strategy --  内存策略
        # CogVideoX-5B is ~19.6 GB, leaving <5 GB free on a 24 GB card.
        # CogVideoX-5B 约占用 19.6 GB，在 24 GB 显卡上剩余不到 5 GB。
        # enable_model_cpu_offload releases components when not in use so that
        # the extra VRAM required during the forward pass can be accommodated.
        # enable_model_cpu_offload 在不使用时释放组件，以便为前向传播所需的额外 VRAM 留出空间。
        if self.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        # Enable VAE tiling to reduce decoder memory peak.
        # 启用 VAE 切片以降低解码器内存峰值。
        # The VAE decoder is the largest single memory spike during inference.
        # VAE 解码器是推理过程中最大的单次内存峰值来源。
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        
        # Enable VAE slicing if available (further reduces memory at slight speed cost).
        # 启用 VAE 分片（如果可用），进一步降低内存占用，但会略微降低速度。
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        
        self._pipe = pipe
        return pipe

    def unload_model(self):
        """Remove model from GPU.
        从 GPU 上卸载模型。"""
        self._pipe = None
        super().unload_model()

    # -- Core inference ---------------------------------------------------
    # -- 核心推理 ---------------------------------------------------

    def _generate_frames(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Run CogVideoX inference and return RGB frame list.
        运行 CogVideoX 推理并返回 RGB 帧列表。

        Args:
            prompt: Text description.
                    文本描述。
            image: Path to conditioning image (I2V mode). If None, uses T2V.
                   条件图像路径（I2V 模式）。如果为 None，则使用 T2V。
            **kwargs: Overrides for num_frames, guidance_scale, seed, etc.
                      用于覆盖 num_frames、guidance_scale、seed 等参数。
                      Also supports:
                      同时支持：
                      - duration (float): desired video seconds (overrides num_frames)
                                          期望的视频秒数（覆盖 num_frames）
                      - resolution (str): e.g. "720P" → (1280, 720)
                                          例如 "720P" → (1280, 720)
                      - fps (int): used with duration to compute num_frames
                                    与 duration 一起用于计算 num_frames

        Returns:
            List of (H, W, 3) uint8 arrays in RGB order.
            (H, W, 3) uint8 数组列表，按 RGB 顺序排列。
        """
        pipe = self._pipe
        if pipe is None:
            raise RuntimeError("Pipeline not loaded — call _load_model() first")

        # ---- resolve parameters -----------------------------------------
        # ---- 解析参数 -----------------------------------------
        # 1) duration → num_frames (if provided, takes precedence)
        # 1) duration → num_frames（如果提供，优先级更高）
        fps = kwargs.get("fps", 24)
        duration = kwargs.get("duration")
        if duration is not None:
            num_frames = max(1, int(duration * fps))
        else:
            num_frames = kwargs.get("num_frames", DEFAULT_NUM_FRAMES)

        # During warm-up / small test, clamp to a safe value
        # 预热/小规模测试期间，限制到安全值
        # (remove after confirming the pipeline works)
        # （确认管线工作正常后可移除此行）
        num_frames = min(num_frames, 13)   # ~0.5 s at 24 fps  ~0.5 秒 @ 24 fps

        guidance_scale = kwargs.get("guidance_scale", 6.0)
        seed = kwargs.get("seed", None)

        # 2) resolution → target (W, H)
        # 2) resolution → 目标 (W, H)
        resolution = kwargs.get("resolution")
        target_size = None
        if resolution is not None:
            res_map = {
                "720P": (1280, 720),
                "720p": (1280, 720),
                "1080P": (1920, 1080),
                "1080p": (1920, 1080),
            }
            target_size = res_map.get(str(resolution).upper())

        # Prepare optional generator
        # 准备可选的生成器
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Load image if provided (I2V mode)
        # 如果提供了图像，加载图像（I2V 模式）
        # CogVideoXImageToVideoPipeline requires an image — T2V is not
        # supported with this pipeline. Fall back to a black canvas if
        # no image is supplied so the pipeline can still run.
        # CogVideoXImageToVideoPipeline 需要图像 — 此管线不支持 T2V。
        # 如果未提供图像，则回退到黑色画布，以便管线仍然可以运行。
        pil_image = None
        if image is not None and os.path.isfile(image):
            pil_image = Image.open(image).convert("RGB")
            if target_size is not None:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Default native resolution: 1360x768, or use target_size if given
            # 默认原生分辨率：1360x768，如果指定了 target_size 则使用它
            size = target_size or (1360, 768)
            pil_image = Image.new("RGB", size, color="black")

        # ---- run pipeline -----------------------------------------------
        # ---- 运行管线 -----------------------------------------------
        # Filter out keys that are not recognised by the diffusers pipeline
        # 过滤掉 diffusers 管线无法识别的键
        excluded_keys = {"num_frames", "guidance_scale", "seed", "fps", "duration", "resolution"}
        with torch.inference_mode():
            output = pipe(  # type: ignore[operator]
                prompt=prompt,
                image=pil_image,
                num_videos_per_prompt=1,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=generator,
                **{k: v for k, v in kwargs.items() if k not in excluded_keys},
            )

        # Each frame is a PIL Image.
        # 每帧都是 PIL 图像。
        # output.frames is a list of lists: [[frame1, frame2, ...]]
        # output.frames 是一个列表的列表：[[frame1, frame2, ...]]
        frames_pil = list(output.frames[0])  # type: ignore[union-attr]

        # Convert PIL → numpy RGB
        # 将 PIL 转换为 numpy RGB 格式
        frames_np = [np.asarray(f, dtype=np.uint8) for f in frames_pil]
        return frames_np

    # -- Prompt building (same pattern as wan.py / seedance.py) -----------
    # -- 提示词构建（与 wan.py / seedance.py 相同模式） -----------

    def _build_turn_prompt(self, case: Dict[str, Any], interaction: Dict[str, Any], turn_index: int) -> str:
        perspective = case.get("settings", {}).get("perspective", "first_person")
        return build_turn_prompt(case, interaction, perspective=perspective, is_first_turn=(turn_index == 0))

    # -- Convenience ------------------------------------------------------
    # -- 便利方法 ------------------------------------------------------

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["model_id"] = self.model_id
        info["torch_dtype"] = str(self.torch_dtype)
        info["pipeline"] = "CogVideoXImageToVideoPipeline"
        return info