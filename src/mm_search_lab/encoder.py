"""OpenCLIP text and image encoders for Day 2 embedding extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union
import warnings

import numpy as np
from PIL import Image


warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")


PathLike = Union[str, Path]


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a 2D embedding array row by row."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array, got shape {}".format(array.shape))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


class OpenClipEncoder:
    """Thin wrapper around open_clip for text/image embedding extraction."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenClipEncoder requires torch and open_clip. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("cuda requested but unavailable; falling back to cpu")
            self.device = "cpu"

        self._torch = torch
        self._open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")

        outputs: List[np.ndarray] = []
        with self._torch.no_grad():
            for batch in _batched(list(texts), self.batch_size):
                tokens = self.tokenizer(batch).to(self.device)
                features = self.model.encode_text(tokens)
                outputs.append(features.float().cpu().numpy())

        embeddings = np.vstack(outputs).astype(np.float32)
        return l2_normalize(embeddings) if self.normalize else embeddings

    def encode_images(self, image_paths: Sequence[PathLike]) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must not be empty")

        outputs: List[np.ndarray] = []
        with self._torch.no_grad():
            for batch in _batched(list(image_paths), self.batch_size):
                images = [self._load_image(path) for path in batch]
                tensor = self._torch.stack(images).to(self.device)
                features = self.model.encode_image(tensor)
                outputs.append(features.float().cpu().numpy())

        embeddings = np.vstack(outputs).astype(np.float32)
        return l2_normalize(embeddings) if self.normalize else embeddings

    def _load_image(self, image_path: PathLike):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError("image file not found: {}".format(path))
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))


def _batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
