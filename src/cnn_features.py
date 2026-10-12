"""Features de imagen para el modelo CNN.

Una ResNet18 preentrenada en ImageNet (torchvision) sin su última capa: cada frame se
convierte en un vector de 512 números. El clasificador de notebooks/07_modelo_cnn.ipynb
se entrena sobre esos vectores, y la app usa `extract_cnn_features()` para preprocesar
la imagen que sube el usuario. Si se cambia algo aquí, hay que volver a correr el 07.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

N_FEATURES = 512
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


@lru_cache(maxsize=1)
def _load():
    """Carga la red una sola vez (la primera vez descarga ~45 MB de pesos)."""
    model = resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()  # se queda con el vector antes de clasificar
    model.eval()
    return model, WEIGHTS.transforms()


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("imagen vacía")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # la red se entrenó con RGB
    return torch.from_numpy(rgb).permute(2, 0, 1)  # HWC uint8 -> CHW


@torch.inference_mode()
def extract_cnn_features_batch(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    """Varios frames BGR -> matriz (n, 512)."""
    model, preprocess = _load()
    batch = torch.stack([preprocess(_to_tensor(img)) for img in imgs_bgr])
    return model(batch).numpy().astype(np.float32)


def extract_cnn_features(img_bgr: np.ndarray) -> np.ndarray:
    """Vector de 512 features de un frame BGR (como lo devuelve cv2.imread)."""
    return extract_cnn_features_batch([img_bgr])[0]
