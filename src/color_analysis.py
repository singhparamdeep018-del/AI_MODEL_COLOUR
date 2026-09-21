"""
Micro-Mend AI - Color Analysis Engine
======================================
Refactored from the original prototype scripts (analyze_color.py,
micro_mend_engine.py, test_real_image.py). This module contains the
colorimetric logic that is common to every prediction pathway:

    - background/noise suppression (color masking)
    - RGB / HSV statistic extraction
    - dominant-hue histogram detection
    - rule-based hue -> color-name classification
    - color-name -> Micro-Mend prototype state mapping
    - the 24-dimensional RGB+HSV feature vector used to train the
      Random Forest model (must stay in sync with color_ml_model.py)

No behavior from the original prototype was changed - thresholds,
mask logic, and the classification boundaries are preserved exactly
as they were validated in the original scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "S1_yellow",
    "S2_yellow_green",
    "S3_green",
    "S4_blue_green",
    "S5_blue",
    "S6_purple",
]

# Human-readable label for each prototype state
STATE_LABELS = {
    "S1_yellow": "Yellow",
    "S2_yellow_green": "Yellow-Green",
    "S3_green": "Green",
    "S4_blue_green": "Blue-Green",
    "S5_blue": "Blue",
    "S6_purple": "Purple",
}

# Approximate swatch color for each state (for the UI color-preview chip)
STATE_SWATCH = {
    "S1_yellow": "#F2C230",
    "S2_yellow_green": "#A8C93B",
    "S3_green": "#3FA65A",
    "S4_blue_green": "#22A6A0",
    "S5_blue": "#3B7DDB",
    "S6_purple": "#8B5FBF",
}

# Rule-based color name -> Micro-Mend prototype state
# (identical mapping used in micro_mend_engine.py / test_real_image.py)
COLOR_TO_STATE = {
    "Yellow": "S1_yellow",
    "Green": "S3_green",
    "Blue-Green": "S4_blue_green",
    "Blue": "S5_blue",
    "Purple": "S6_purple",
    "Orange": "S1_yellow",
    "Red": "S2_yellow_green",
    "Magenta": "S6_purple",
}

FEATURE_NAMES = [
    "R_mean", "G_mean", "B_mean",
    "R_std", "G_std", "B_std",
    "R_median", "G_median", "B_median",
    "H_mean", "S_mean", "V_mean",
    "H_std", "S_std", "V_std",
    "H_median", "S_median", "V_median",
    "H_min", "S_min", "V_min",
    "H_max", "S_max", "V_max",
]

SATURATION_THRESHOLD = 40
BRIGHTNESS_THRESHOLD = 40
MIN_COLOR_PIXELS = 100


@dataclass
class ColorReading:
    """Container for the colorimetric measurements of one image."""

    width: int
    height: int
    r_mean: float
    g_mean: float
    b_mean: float
    h_median: float
    s_median: float
    v_median: float
    dominant_hue: int
    colored_pixel_count: int
    color_name: str
    color_state: str
    feature_vector: np.ndarray = field(repr=False)


def build_color_mask(hsv_image: np.ndarray) -> np.ndarray:
    """Boolean mask that keeps pixels with meaningful color, discarding
    near-white / near-black background noise."""
    saturation = hsv_image[:, :, 1]
    brightness = hsv_image[:, :, 2]
    return (saturation > SATURATION_THRESHOLD) & (brightness > BRIGHTNESS_THRESHOLD)


def classify_color(h: float, s: float, v: float) -> str:
    """Rule-based hue/sat/val -> color-name classifier.
    Preserved exactly from analyze_color.py / micro_mend_engine.py."""
    if s < SATURATION_THRESHOLD:
        if v < 60:
            return "Black / Very Dark"
        elif v > 200:
            return "White / Very Light"
        else:
            return "Gray"

    if h < 8 or h >= 170:
        return "Red"
    elif h < 22:
        return "Orange"
    elif h < 40:
        return "Yellow"
    elif h < 80:
        return "Green"
    elif h < 105:
        return "Blue-Green"
    elif h < 125:
        return "Blue"
    elif h < 165:
        return "Purple"
    else:
        return "Magenta"


def _build_feature_vector(rgb_pixels: np.ndarray, hsv_pixels: np.ndarray) -> np.ndarray:
    """24-dimensional RGB+HSV feature vector, identical layout to the one
    used to train micromend_color_ml_model.pkl (color_ml_model.py)."""
    r, g, b = rgb_pixels[:, 0], rgb_pixels[:, 1], rgb_pixels[:, 2]
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]

    features = [
        np.mean(r), np.mean(g), np.mean(b),
        np.std(r), np.std(g), np.std(b),
        np.median(r), np.median(g), np.median(b),
        np.mean(h), np.mean(s), np.mean(v),
        np.std(h), np.std(s), np.std(v),
        np.median(h), np.median(s), np.median(v),
        np.min(h), np.min(s), np.min(v),
        np.max(h), np.max(s), np.max(v),
    ]
    return np.array(features, dtype=np.float32)


def analyze_image_colors(image_bgr: np.ndarray) -> ColorReading:
    """Run the full colorimetric analysis on a BGR (OpenCV-style) image
    array and return a ColorReading with everything downstream code needs:
    RGB/HSV stats, dominant hue, rule-based color name + prototype state,
    and the ML feature vector for the Random Forest model.
    """
    height, width = image_bgr.shape[:2]

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = build_color_mask(hsv_image)

    if np.sum(mask) < MIN_COLOR_PIXELS:
        # Fall back to the whole image if too little "color" survives
        # the mask (e.g. very pale or very small sample regions).
        rgb_pixels = rgb_image.reshape(-1, 3)
        hsv_pixels = hsv_image.reshape(-1, 3)
    else:
        rgb_pixels = rgb_image[mask]
        hsv_pixels = hsv_image[mask]

    r_mean, g_mean, b_mean = np.mean(rgb_pixels, axis=0)

    h_values = hsv_pixels[:, 0]
    s_median = float(np.median(hsv_pixels[:, 1]))
    v_median = float(np.median(hsv_pixels[:, 2]))
    h_median = float(np.median(h_values))

    histogram = np.bincount(h_values, minlength=180)
    dominant_hue = int(np.argmax(histogram))

    color_name = classify_color(dominant_hue, s_median, v_median)
    color_state = COLOR_TO_STATE.get(color_name, "Unknown")

    feature_vector = _build_feature_vector(rgb_pixels, hsv_pixels)

    return ColorReading(
        width=width,
        height=height,
        r_mean=float(r_mean),
        g_mean=float(g_mean),
        b_mean=float(b_mean),
        h_median=h_median,
        s_median=s_median,
        v_median=v_median,
        dominant_hue=dominant_hue,
        colored_pixel_count=int(len(hsv_pixels)),
        color_name=color_name,
        color_state=color_state,
        feature_vector=feature_vector,
    )
