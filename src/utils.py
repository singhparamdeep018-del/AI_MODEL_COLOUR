"""
Micro-Mend AI - Utility Helpers
================================
Small, dependency-light helpers for safely turning an uploaded file
(Streamlit's UploadedFile, a path, or raw bytes) into a validated
OpenCV BGR image array, plus a couple of formatting helpers used by
the UI layer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG"}
MAX_DIMENSION = 4000  # guard against absurdly large uploads


class ImageLoadError(Exception):
    """Raised when an uploaded file cannot be safely turned into an image."""


def load_image_from_upload(uploaded_file) -> Tuple[np.ndarray, Image.Image]:
    """Validate and decode a Streamlit UploadedFile.

    Returns a tuple of:
        - bgr_image: np.ndarray (H, W, 3) in BGR order, ready for OpenCV
        - pil_image: PIL.Image in RGB, ready for st.image() display

    Raises ImageLoadError with a user-friendly message on any failure.
    """
    if uploaded_file is None:
        raise ImageLoadError("No file was provided.")

    try:
        pil_image = Image.open(uploaded_file)
        pil_image.load()
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            "This file could not be read as an image. "
            "Please upload a valid PNG or JPG/JPEG file."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any decode failure cleanly
        raise ImageLoadError(f"Could not open the uploaded file ({exc}).") from exc

    fmt = (pil_image.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported image format '{fmt or 'unknown'}'. "
            "Please upload a PNG or JPG/JPEG file."
        )

    pil_image = pil_image.convert("RGB")

    width, height = pil_image.size
    if width == 0 or height == 0:
        raise ImageLoadError("The uploaded image has invalid dimensions.")

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageLoadError(
            f"Image is too large ({width}x{height}). "
            f"Please upload an image no larger than {MAX_DIMENSION}px on a side."
        )

    rgb_array = np.array(pil_image)
    bgr_image = rgb_array[:, :, ::-1].copy()  # RGB -> BGR for OpenCV-based code

    return bgr_image, pil_image


def format_percent(value: float) -> str:
    """Consistent percentage formatting used throughout the UI."""
    return f"{value:.1f}%"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
