"""Carga de datos, modelos y paleta compartida por la app."""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURES, extract_features  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"

CLASSES = ["bg", "play", "challenge", "throwin"]
CLASS_LABELS = {
    "bg": "bg (sin evento)",
    "play": "play (pase)",
    "challenge": "challenge (disputa)",
    "throwin": "throwin (saque de banda)",
}
# Un color fijo por clase. Validado para daltonismo (deutan/protan/tritan) con
# separación ΔE >= 6 en todos los pares, en fondo claro y oscuro. Como el par más
# cercano queda en la banda 6-8, el nombre de la clase siempre va escrito junto al color.
CLASS_COLORS = {
    "bg": "#c98500",
    "play": "#2a78d6",
    "challenge": "#e34948",
    "throwin": "#1baf7a",
}

MODEL_NAMES = {
    "dummy": "Dummy (línea base)",
    "logreg": "Regresión Logística",
    "rf": "Random Forest",
    "hgb": "HistGradientBoosting",
    "cnn": "ResNet18 + clasificador",
}
# Tonos para fondo oscuro. El Dummy va en gris a propósito: es la referencia, no un competidor.
MODEL_COLORS = {
    "dummy": "#8a8984",
    "logreg": "#3987e5",
    "rf": "#d95926",
    "hgb": "#199e70",
    "cnn": "#9085e9",
}
_EXTRA_MODEL_COLORS = ["#eda100", "#e87ba4", "#008300"]

FEATURE_LABELS = {
    "brillo": "Brillo (media de gris)",
    "contraste": "Contraste (desv. de gris)",
    "saturacion": "Saturación (media S en HSV)",
    "green_ratio": "Proporción de verde (césped)",
    "edge_density": "Densidad de bordes (Canny)",
}


def model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key)


def model_color_map(keys) -> dict:
    """Color por modelo; los modelos desconocidos toman los slots libres en orden fijo."""
    out, extra = {}, iter(_EXTRA_MODEL_COLORS)
    for k in sorted(keys, key=_model_order):
        out[model_name(k)] = MODEL_COLORS.get(k) or next(extra, "#52514e")
    return out


def _model_order(key: str) -> int:
    order = list(MODEL_NAMES)
    return order.index(key) if key in order else len(order)


def sort_models(keys) -> list:
    return sorted(keys, key=_model_order)


# ---------------------------------------------------------------- datos

@st.cache_data
def load_frame_metrics() -> pd.DataFrame:
    df = pd.read_csv(PROC / "frame_metrics.csv")
    df["existe"] = [(ROOT / p).exists() for p in df.path]
    return df


@st.cache_data
def load_cnn_pca() -> pd.DataFrame:
    """Vectores de la ResNet18 reducidos a 2D con PCA (los genera notebooks/07_modelo_cnn.ipynb)."""
    f = PROC / "cnn_pca.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


@st.cache_data
def load_results(kind: str) -> pd.DataFrame:
    """Concatena todos los results/<kind>_*.csv (metrics o predictions)."""
    frames = []
    for f in sorted(RESULTS_DIR.glob(f"{kind}_*.csv")):
        d = pd.read_csv(f)
        d["origen"] = f.stem.split("_", 1)[1]
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["modelo"] = df.model.map(model_name)
    return df


# -------------------------------------------------------------- modelos

@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Carga todo lo que haya en models/*.joblib. Devuelve {clave: modelo}."""
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in sorted(MODELS_DIR.glob("*.joblib")):
            try:
                models[f.stem] = joblib.load(f)
            except Exception as e:  # un modelo roto no debe tumbar la app
                st.warning(f"No se pudo cargar {f.name}: {e}")
    return dict(sorted(models.items(), key=lambda kv: _model_order(kv[0])))


def is_cnn(key: str, model) -> bool:
    """Los modelos clásicos se entrenaron con un DataFrame de las 5 features;
    cualquier otro (el de D) recibe el vector de la ResNet."""
    names = getattr(model, "feature_names_in_", None)
    return key.startswith("cnn") or names is None or list(names) != FEATURES


@st.cache_resource(show_spinner=False)
def _cnn_extractor():
    """Función de src/cnn_features.py que convierte un frame BGR en un vector."""
    mod = importlib.import_module("src.cnn_features")
    for name in ("extract_cnn_features", "extract_features", "extract_embedding", "embed"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError("src/cnn_features.py no expone extract_cnn_features(img_bgr)")


def predict_all(img_bgr: np.ndarray, keys: list, models: dict) -> pd.DataFrame:
    """Predice con cada modelo pedido. Una fila por modelo con pred y proba_<clase>."""
    feats = pd.DataFrame([extract_features(img_bgr)])[FEATURES]
    cnn_vec = None
    rows = []
    for k in keys:
        m = models[k]
        try:
            if is_cnn(k, m):
                if cnn_vec is None:
                    cnn_vec = np.asarray(_cnn_extractor()(img_bgr), dtype=np.float32).reshape(1, -1)
                X = cnn_vec
            else:
                X = feats
            proba = m.predict_proba(X)[0]
            row = {"model": k, "pred": str(m.classes_[int(np.argmax(proba))])}
            row.update({f"proba_{c}": float(p) for c, p in zip(m.classes_, proba)})
        except Exception as e:
            row = {"model": k, "pred": None, "error": str(e)}
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in CLASSES:
        if f"proba_{c}" not in out:
            out[f"proba_{c}"] = np.nan
    out["modelo"] = out.model.map(model_name)
    return out
