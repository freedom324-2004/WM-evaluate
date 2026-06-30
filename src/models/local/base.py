"""
LocalVideoModel — intermediate abstract base for local (self-hosted) video generation models.
LocalVideoModel — 本地（自托管）视频生成模型的中间抽象基类。

Subclasses only need to implement:
子类只需实现：
    _load_model() -> model object (cached after first call)
    _load_model() -> 模型对象（首次调用后缓存）
    _generate_frames(prompt, image=None, **kwargs) -> List[np.ndarray]
    _generate_frames(prompt, image=None, **kwargs) -> List[np.ndarray]

The base class handles:
基类负责处理：
    - GPU / CPU device selection
      GPU / CPU 设备选择
    - Lazy model loading (self._model cache)
      延迟模型加载（self._model 缓存）
    - Frames → MP4 encoding (_save_frames)
      帧 → MP4 编码（_save_frames）
    - Turning _generate_frames output into the standard {"code", "video_path"} dict
      将 _generate_frames 输出转换为标准的 {"code", "video_path"} 字典
"""
from abc import abstractmethod
from typing import Optional, Dict, Any, List
import os
import tempfile

import numpy as np
import cv2

from ..base import BaseVideoModel


class LocalVideoModel(BaseVideoModel):
    """Abstract base for locally-run video models.
    本地运行视频模型的抽象基类。

    Subclasses implement ``_load_model`` and ``_generate_frames``;
    everything else (video encoding, device management, lazy loading)
    is handled here.
    子类实现 ``_load_model`` 和 ``_generate_frames``；
    其他所有功能（视频编码、设备管理、延迟加载）均在此处理。
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name)
        self.device = device if self._is_cuda_available() else "cpu"
        self._model = None          # lazy-loaded pipeline  延迟加载的管线
        self._loaded = False

    # ------------------------------------------------------------------
    # Subclass API  (must override)  子类 API（必须重写）
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(self):
        """Load model weights into memory.
        将模型权重加载到内存中。

        Called **once** on the first ``generate()`` call; the result is
        cached in ``self._model``.  Implementations may use any framework
        (diffusers, custom PyTorch, etc.).
        在首次 ``generate()`` 调用时**仅调用一次**；结果会
        缓存到 ``self._model`` 中。实现可以使用任何框架
        （diffusers、自定义 PyTorch 等）。
        """
        ...

    @abstractmethod
    def _generate_frames(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Core inference routine.
        核心推理例程。

        Args:
            prompt: Text description.
                    文本描述。
            image: Path to conditioning image (or None for T2V mode).
                   条件图像路径（T2V 模式下为 None）。
            **kwargs: Additional model-specific parameters.
                      其他模型特定参数。

        Returns:
            List of (H, W, 3) uint8 numpy arrays (RGB order).
            (H, W, 3) uint8 numpy 数组列表（RGB 顺序）。
        """
        ...

    # ------------------------------------------------------------------
    # Base implementation  (inherited by all local models)
    # 基类实现（所有本地模型继承）
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a single video segment.
        生成单个视频片段。

        Loads model on first call, runs inference, encodes frames to MP4.
        首次调用时加载模型，运行推理，将帧编码为 MP4。
        """
        try:
            if not self._loaded:
                self._model = self._load_model()
                self._loaded = True

            # Strip multi-turn bookkeeping kwargs before forwarding
            # 在转发之前剥离多轮记账关键字参数
            meta_keys = {"turn", "action", "interaction_type", "output_path"}
            # Extract output_path if provided, so _save_frames can use it
            # 提取 output_path（如果有），以便 _save_frames 使用
            output_path = kwargs.get("output_path", None)
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in meta_keys}

            frames = self._generate_frames(prompt, image=image, **gen_kwargs)
            if not frames:
                return {"code": -1, "error": "No frames returned from model"}

            video_path = self._save_frames(frames, fps=kwargs.get("fps", 24), output_path=output_path)
            return {"code": 0, "video_path": video_path}

        except Exception as e:
            return {"code": -1, "error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # Helpers  辅助方法
    # ------------------------------------------------------------------

    def _save_frames(
        self,
        frames: List[np.ndarray],
        fps: int = 24,
        output_path: Optional[str] = None,
    ) -> str:
        """Encode a list of RGB frames into an MP4 file.
        将 RGB 帧列表编码为 MP4 文件。

        Args:
            frames: List of (H, W, 3) uint8 arrays (RGB order).
                     (H, W, 3) uint8 数组列表（RGB 顺序）。
            fps: Frames per second.
                 每秒帧数。
            output_path: If None, a temporary file is created.
                         如果为 None，则创建临时文件。

        Returns:
            Absolute path to the generated MP4 file.
            生成的 MP4 文件的绝对路径。
        """
        if output_path is None:
            tmpdir = tempfile.mkdtemp(prefix="wm_local_")
            output_path = os.path.join(tmpdir, "output.mp4")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

        for frame in frames:
            # Convert RGB → BGR for cv2
            # 将 RGB 转换为 BGR 以适配 cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

        return os.path.abspath(output_path)

    def unload_model(self):
        """Release model from GPU memory.
        从 GPU 内存中释放模型。"""
        self._model = None
        self._loaded = False
        if "cuda" in self.device:
            import torch
            torch.cuda.empty_cache()

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # -- informational ---------------------------------------------------
    # -- 信息 ---------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info["device"] = self.device
        info["loaded"] = self._loaded
        return info

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"loaded={self._loaded})>"
        )