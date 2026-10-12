"""Features de imagen usadas por los modelos clásicos.

Replica exactamente `frame_metrics()` de notebooks/03_eda_visual.ipynb, que generó
data/processed/frame_metrics.csv. Si se cambia algo aquí, los modelos dejan de
recibir lo mismo con lo que se entrenaron.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURES = ["brillo", "contraste", "saturacion", "green_ratio", "edge_density"]

GREEN_LO = np.array([35, 40, 40])
GREEN_HI = np.array([85, 255, 255])
RESIZE_W = 320


def extract_features(img_bgr: np.ndarray) -> dict:
    """Calcula las 5 features de un frame BGR (como lo devuelve cv2.imread)."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("imagen vacía")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (RESIZE_W, int(RESIZE_W * h / w)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    edges = cv2.Canny(gray, 100, 200)
    return {
        "brillo": float(gray.mean()),
        "contraste": float(gray.std()),
        "saturacion": float(hsv[..., 1].mean()),
        "green_ratio": float((green > 0).mean()),
        "edge_density": float((edges > 0).mean()),
    }


def decode_image(data: bytes) -> np.ndarray:
    """Bytes de un archivo de imagen -> array BGR, igual que cv2.imread."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("no se pudo leer la imagen")
    return img
