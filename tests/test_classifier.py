"""Tests for preprocessing, labels and archives (no Home Assistant needed)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from custom_components.snapshot_classifier.classifier import (
    Classifier,
    CropBox,
    ModelArchiveError,
    crop_image,
    load_model,
    parse_labels,
    preprocess,
    read_model_archive,
    save_model,
)

from .conftest import LABELS, make_zip


def test_parse_labels() -> None:
    """Teachable Machine's 'index name' format, spaces in names, blank lines."""
    assert parse_labels("0 Jeep\n1 Altre auto\n\n2 Vuoto\n") == ["Jeep", "Altre auto", "Vuoto"]
    assert parse_labels("﻿cat\r\ndog\r\n") == ["﻿cat", "dog"]
    assert parse_labels("10 x\n") == ["x"]


def test_crop_box() -> None:
    """Normalized boxes convert to pixels and validate."""
    box = CropBox(0.25, 0.5, 0.5, 0.5)
    assert box.pixels(656, 368) == (164, 184, 492, 368)
    assert box.is_valid
    assert not CropBox(0.6, 0, 0.5, 1).is_valid
    assert not CropBox(0, 0, 0, 1).is_valid
    assert CropBox().is_full_image
    assert CropBox.from_mapping(box.as_dict()) == box


def test_preprocess_center_square() -> None:
    """Centre square, resize to the model size, scale to [-1, 1]."""
    image = Image.new("RGB", (300, 100), (0, 0, 0))
    image.paste((255, 255, 255), (100, 0, 200, 100))  # the centre square is white
    data = preprocess(image, 224, 224)
    assert data.shape == (1, 224, 224, 3)
    assert data.dtype == np.float32
    assert np.allclose(data, 1.0)

    black = preprocess(Image.new("RGB", (10, 40)), 32, 32)
    assert np.allclose(black, -1.0)


def test_crop_before_square() -> None:
    """The crop box is applied before the centre square."""
    image = Image.new("RGB", (400, 200), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 100, 100))
    cropped = crop_image(image, CropBox(0, 0, 0.25, 0.5))
    assert cropped.size == (100, 100)
    assert np.allclose(preprocess(cropped, 8, 8)[..., 0], 1.0)


def test_archive_roundtrip(tmp_path: Path, model_bytes: bytes) -> None:
    """Read an export archive, store it, load it back and classify."""
    archive = make_zip(tmp_path / "m.zip", model_bytes, LABELS)
    classifier = read_model_archive(archive)
    assert classifier.labels == LABELS
    assert classifier.input_size == (64, 64)

    save_model(tmp_path / "store", classifier)
    loaded = load_model(tmp_path / "store")
    assert loaded.sha256 == classifier.sha256

    buffer = io.BytesIO()
    Image.new("RGB", (656, 368), (50, 100, 150)).save(buffer, format="JPEG")
    result = loaded.classify(buffer.getvalue(), CropBox(0.1, 0.1, 0.5, 0.8))
    assert list(result) == LABELS
    assert sum(result.values()) == pytest.approx(1.0)


def test_archive_in_subfolder_with_other_model_name(tmp_path: Path, model_bytes: bytes) -> None:
    """Members are found by file name, a single .tflite with any name is accepted."""
    import zipfile

    path = tmp_path / "m.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("export/my_model.tflite", model_bytes)
        archive.writestr("export/labels.txt", "a\nb\nc\n")
    assert read_model_archive(path).labels == ["a", "b", "c"]


def test_duplicate_labels(model_bytes: bytes) -> None:
    """Duplicate classes cannot be enum options."""
    with pytest.raises(ModelArchiveError) as err:
        Classifier(model_bytes, ["a", "a", "b"])
    assert err.value.reason == "invalid_labels"


def test_undecodable_image(model_bytes: bytes) -> None:
    """Garbage camera bytes raise ValueError."""
    with pytest.raises(ValueError):
        Classifier(model_bytes, LABELS).classify(b"nope", CropBox())
