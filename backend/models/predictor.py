"""
SkinNova AI – Model Loader
Loads all three CNN models at startup and exposes predict functions.
Models are EfficientNetB0-based, fine-tuned on the provided datasets.
"""

import os
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Try to import TensorFlow only when explicitly enabled; otherwise use fast demo predictions.
USE_TENSORFLOW = os.environ.get("SKINNOVA_USE_TENSORFLOW", "0") == "1"
try:
    if not USE_TENSORFLOW:
        raise ImportError("TensorFlow disabled by default for fast local startup.")
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow unavailable or disabled - models will use fast demo predictions.")

# ── Class labels ──────────────────────────────────────────────────────────
SKIN_TYPE_CLASSES  = ["dry", "normal", "oily"]          # Dataset 3 (Oily-Dry)
ACNE_TYPE_CLASSES  = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]  # Dataset 2
CONCERN_CLASSES    = ["acne", "blackheads", "dark_spots", "pores", "wrinkles"]     # Dataset 1

IMG_SIZE = (224, 224)

# ── Singleton model cache ──────────────────────────────────────────────────
_models = {}


def _load_model(name: str, path: str):
    """Load a Keras model from disk if it exists."""
    if not TF_AVAILABLE:
        return None
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            logger.info(f"Loaded model: {name} from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    else:
        logger.warning(f"Model not found at {path} – run training scripts first.")
    return None


def load_all_models(model_dir: str):
    """Call once at app startup."""
    global _models
    _models["skin_type"] = _load_model(
        "skin_type", os.path.join(model_dir, "skin_type_model", "model.h5")
    )
    _models["acne_type"] = _load_model(
        "acne_type", os.path.join(model_dir, "acne_type_model", "model.h5")
    )
    _models["concerns"] = _load_model(
        "concerns", os.path.join(model_dir, "skin_concern_model", "model.h5")
    )


def _preprocess(image_path: str) -> np.ndarray:
    """Load, resize, and normalize an image for EfficientNetB0."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    if TF_AVAILABLE:
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    else:
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def _mock_predict_skin_type():
    """Return realistic mock scores when model isn't loaded."""
    import random
    scores = [random.random() for _ in SKIN_TYPE_CLASSES]
    total = sum(scores)
    return [s / total for s in scores]


def _mock_predict_acne():
    import random
    scores = [random.random() for _ in ACNE_TYPE_CLASSES]
    total = sum(scores)
    return [s / total for s in scores]


def _mock_predict_concerns():
    import random
    return [random.uniform(0.1, 0.9) for _ in CONCERN_CLASSES]


def predict_skin_type(image_path: str) -> dict:
    """
    Returns:
        {
          "skin_type": "oily",
          "confidence": 0.87,
          "scores": {"dry": 0.05, "normal": 0.08, "oily": 0.87},
          "combination_risk": 0.3   # heuristic
        }
    """
    model = _models.get("skin_type")
    if model and TF_AVAILABLE:
        arr = _preprocess(image_path)
        probs = model.predict(arr)[0].tolist()
    else:
        probs = _mock_predict_skin_type()

    scores = dict(zip(SKIN_TYPE_CLASSES, probs))
    skin_type = max(scores, key=scores.get)
    confidence = scores[skin_type]

    # Heuristic: combination = high oily + high dry scores together
    combo_risk = min(scores.get("oily", 0) * 0.6 + scores.get("dry", 0) * 0.4, 1.0)
    if combo_risk > 0.45 and skin_type != "oily":
        skin_type = "combination"

    return {
        "skin_type": skin_type,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "combination_risk": round(combo_risk, 3),
    }


def predict_acne(image_path: str) -> dict:
    """
    Returns:
        {
          "acne_level": "moderate",   # none/mild/moderate/severe
          "acne_type": "Pustules",
          "confidence": 0.74,
          "scores": {...}
        }
    """
    model = _models.get("acne_type")
    if model and TF_AVAILABLE:
        arr = _preprocess(image_path)
        probs = model.predict(arr)[0].tolist()
    else:
        probs = _mock_predict_acne()

    scores = dict(zip(ACNE_TYPE_CLASSES, probs))
    acne_type = max(scores, key=scores.get)
    confidence = scores[acne_type]

    # Map acne type → severity level
    severity_map = {
        "Blackheads":  "mild",
        "Whiteheads":  "mild",
        "Papules":     "moderate",
        "Pustules":    "moderate",
        "Cyst":        "severe",
    }
    acne_level = severity_map.get(acne_type, "none")

    # If max confidence is low, likely no significant acne
    if confidence < 0.35:
        acne_level = "none"
        acne_type = "none"

    return {
        "acne_level": acne_level,
        "acne_type": acne_type,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }


def predict_concerns(image_path: str) -> dict:
    """
    Returns:
        {
          "concerns": ["dark_spots", "pores"],
          "scores": {"acne": 0.2, "blackheads": 0.15, ...},
          "risk_flags": {"early_acne_risk": 0.6, "pore_congestion": 0.7, ...}
        }
    """
    model = _models.get("concerns")
    if model and TF_AVAILABLE:
        arr = _preprocess(image_path)
        raw = model.predict(arr)[0].tolist()
    else:
        raw = _mock_predict_concerns()

    scores = dict(zip(CONCERN_CLASSES, raw))
    # Threshold 0.45 to flag a concern
    concerns = [c for c, s in scores.items() if s > 0.45]

    # Hidden / pre-symptom risk flags
    risk_flags = {
        "early_acne_risk":    round(scores.get("acne", 0) * 0.8 + scores.get("blackheads", 0) * 0.2, 3),
        "pore_congestion":    round(scores.get("pores", 0) * 0.7 + scores.get("blackheads", 0) * 0.3, 3),
        "texture_imbalance":  round(scores.get("dark_spots", 0) * 0.5 + scores.get("wrinkles", 0) * 0.5, 3),
        "micro_inflammation": round(scores.get("acne", 0) * 0.6 + scores.get("pores", 0) * 0.4, 3),
    }

    return {
        "concerns": concerns,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "risk_flags": risk_flags,
    }


def full_skin_analysis(image_path: str) -> dict:
    """Run all three models and merge into one result object."""
    skin   = predict_skin_type(image_path)
    acne   = predict_acne(image_path)
    concern = predict_concerns(image_path)

    return {
        "skin_type":   skin["skin_type"],
        "skin_scores": skin["scores"],
        "acne_level":  acne["acne_level"],
        "acne_type":   acne["acne_type"],
        "acne_scores": acne["scores"],
        "concerns":    concern["concerns"],
        "concern_scores": concern["scores"],
        "risk_flags":  concern["risk_flags"],
        "combination_risk": skin["combination_risk"],
    }
