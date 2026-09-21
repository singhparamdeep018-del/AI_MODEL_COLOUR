"""
Micro-Mend AI - Prediction Orchestration
=========================================
Loads the two trained models produced by the original prototype
scripts and combines them with the rule-based color engine, following
the same multi-model comparison strategy validated in
test_real_image.py.

Model roles (decided after inspecting the prototype scripts):

  1. RGB + HSV Random Forest (micromend_color_ml_model.pkl)
     -> PRIMARY model. This is a colorimetric prototype first and
        foremost, and the Random Forest operates directly on RGB/HSV
        statistics, which is the most defensible signal for a
        color-responsive hydrogel sensor. It also natively exposes a
        clean 6-class probability distribution for the confidence UI.

  2. MobileNetV2 (micromend_color_model.keras)
     -> EXPERIMENTAL cross-check. Preserved as in the original
        prototype. Useful as a secondary opinion, but the project
        notes observed disagreements with the color engine on real
        (non-synthetic) images, so it is surfaced as a supporting
        signal rather than the primary result.

  3. Rule-based color engine (hue thresholds -> color name -> state)
     -> Deterministic, fully explainable cross-check with no learned
        parameters. Always available even if both model files are
        missing.

The three are combined into an "agreement" score, mirroring the
majority-vote logic in test_real_image.py, so the UI can show whether
the systems concur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import streamlit as st

from src.color_analysis import CLASS_NAMES, ColorReading, analyze_image_colors

RF_MODEL_FILENAME = "micromend_color_ml_model.pkl"
CNN_MODEL_FILENAME = "micromend_color_model.keras"
CNN_IMAGE_SIZE = (224, 224)


@dataclass
class ModelResult:
    """Normalized result shape shared by both models."""

    available: bool
    predicted_state: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    color: ColorReading
    random_forest: ModelResult
    mobilenet: ModelResult
    primary_state: str
    primary_confidence: Optional[float]
    primary_probabilities: Optional[Dict[str, float]]
    agreement_count: int
    agreement_total: int
    votes: Dict[str, str]


@st.cache_resource(show_spinner=False)
def _load_rf_raw(model_path: str):
    import os

    if not os.path.exists(model_path):
        return None, f"Random Forest model file not found: {model_path}"
    try:
        import joblib

        return joblib.load(model_path), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not load Random Forest model: {exc}"


@st.cache_resource(show_spinner=False)
def _load_cnn_raw(model_path: str):
    import os

    if not os.path.exists(model_path):
        return None, f"MobileNetV2 model file not found: {model_path}"
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path), None
    except ModuleNotFoundError:
        return None, "TensorFlow is not installed - the experimental deep-learning model is disabled."
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not load MobileNetV2 model: {exc}"


def load_models(models_dir: str) -> Dict[str, tuple]:
    """Load both models once (Streamlit-cached) and return the raw model
    objects plus any load errors, keyed by model name."""
    import os

    rf_path = os.path.join(models_dir, RF_MODEL_FILENAME)
    cnn_path = os.path.join(models_dir, CNN_MODEL_FILENAME)

    rf_model, rf_error = _load_rf_raw(rf_path)
    cnn_model, cnn_error = _load_cnn_raw(cnn_path)

    return {
        "random_forest": (rf_model, rf_error),
        "mobilenet": (cnn_model, cnn_error),
    }


def predict_random_forest(rf_model, feature_vector: np.ndarray) -> ModelResult:
    if rf_model is None:
        return ModelResult(available=False, error="Random Forest model is not loaded.")

    try:
        probabilities = rf_model.predict_proba(feature_vector.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        prob_dict = {name: float(probabilities[i] * 100) for i, name in enumerate(CLASS_NAMES)}
        return ModelResult(
            available=True,
            predicted_state=CLASS_NAMES[index],
            confidence=float(probabilities[index] * 100),
            probabilities=prob_dict,
        )
    except Exception as exc:  # noqa: BLE001
        return ModelResult(available=False, error=f"Random Forest inference failed: {exc}")


def predict_mobilenet(cnn_model, rgb_image: np.ndarray) -> ModelResult:
    if cnn_model is None:
        return ModelResult(available=False, error="MobileNetV2 model is not loaded.")

    try:
        import cv2

        resized = cv2.resize(rgb_image, CNN_IMAGE_SIZE)
        batch = np.expand_dims(resized, axis=0)
        predictions = cnn_model.predict(batch, verbose=0)[0]
        index = int(np.argmax(predictions))
        prob_dict = {name: float(predictions[i] * 100) for i, name in enumerate(CLASS_NAMES)}
        return ModelResult(
            available=True,
            predicted_state=CLASS_NAMES[index],
            confidence=float(predictions[index] * 100),
            probabilities=prob_dict,
        )
    except Exception as exc:  # noqa: BLE001
        return ModelResult(available=False, error=f"MobileNetV2 inference failed: {exc}")


def run_full_analysis(image_bgr: np.ndarray, models: Dict[str, tuple]) -> AnalysisResult:
    """Run the color engine + both ML models and combine the results,
    mirroring the comparison logic from test_real_image.py."""
    import cv2

    color_reading = analyze_image_colors(image_bgr)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    rf_model, _ = models["random_forest"]
    cnn_model, _ = models["mobilenet"]

    rf_result = predict_random_forest(rf_model, color_reading.feature_vector)
    cnn_result = predict_mobilenet(cnn_model, rgb_image)

    # Primary result: Random Forest when available, otherwise fall back to
    # the deterministic color engine so the app never fails to produce a
    # result even without trained model files.
    if rf_result.available:
        primary_state = rf_result.predicted_state
        primary_confidence = rf_result.confidence
        primary_probabilities = rf_result.probabilities
    else:
        primary_state = color_reading.color_state
        primary_confidence = None
        primary_probabilities = None

    votes = {"Color Engine": color_reading.color_state}
    if rf_result.available:
        votes["RGB + HSV Random Forest"] = rf_result.predicted_state
    if cnn_result.available:
        votes["MobileNetV2 (experimental)"] = cnn_result.predicted_state

    tally: Dict[str, int] = {}
    for state in votes.values():
        tally[state] = tally.get(state, 0) + 1
    agreement_count = max(tally.values()) if tally else 0

    return AnalysisResult(
        color=color_reading,
        random_forest=rf_result,
        mobilenet=cnn_result,
        primary_state=primary_state,
        primary_confidence=primary_confidence,
        primary_probabilities=primary_probabilities,
        agreement_count=agreement_count,
        agreement_total=len(votes),
        votes=votes,
    )
