"""Regression test for the BGR/RGB channel-order bug.

Ultralytics interprets numpy array inputs as BGR. Feeding it an RGB array
makes the model see colour-inverted pixels - high-visibility yellow becomes
blue - which silently destroys vest detection while producing no error.

This test pins the correct behaviour: a BGR array must produce the same
detections as passing the file path or a PIL image.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "models" / "best.pt"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _detections(model, source, conf=0.4):
    result = model(source, conf=conf, verbose=False)[0]
    if result.boxes is None:
        return []
    return sorted(result.names[int(b.cls[0])] for b in result.boxes)


@pytest.fixture(scope="module")
def model():
    ultralytics = pytest.importorskip("ultralytics")
    if not WEIGHTS.exists():
        pytest.skip("models/best.pt not present")
    return ultralytics.YOLO(str(WEIGHTS))


@pytest.fixture(scope="module")
def sample_image():
    images = sorted(FIXTURE_DIR.glob("*.jpg")) + sorted(FIXTURE_DIR.glob("*.jpeg"))
    if not images:
        pytest.skip("no fixture image in tests/fixtures/")
    return images[0]


def test_bgr_array_matches_path(model, sample_image):
    """A BGR array must detect the same classes as the file path."""
    rgb = np.array(Image.open(sample_image).convert("RGB"))
    bgr = rgb[:, :, ::-1]
    assert _detections(model, bgr) == _detections(model, str(sample_image))


def test_bgr_array_matches_pil(model, sample_image):
    """A BGR array must detect the same classes as a PIL image."""
    pil = Image.open(sample_image).convert("RGB")
    bgr = np.array(pil)[:, :, ::-1]
    assert _detections(model, bgr) == _detections(model, pil)


def test_rgb_array_is_the_known_failure_mode(model, sample_image):
    """Documents the bug: an RGB array loses detections a BGR array finds.

    If this test ever fails, ultralytics has changed its input convention -
    revisit the conversion in app/streamlit_app.py.
    """
    rgb = np.array(Image.open(sample_image).convert("RGB"))
    bgr = rgb[:, :, ::-1]
    assert len(_detections(model, bgr)) >= len(_detections(model, rgb))
