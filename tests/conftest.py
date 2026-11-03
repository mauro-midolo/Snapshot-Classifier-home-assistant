"""Fixtures for Snapshot Classifier tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import io
from pathlib import Path
from unittest.mock import patch
import zipfile

from PIL import Image
import pytest

from homeassistant.components.camera import Image as CameraImage

from .tflite_builder import build_teachable_machine_like

LABELS = ["Mine", "Other", "Empty"]


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations in all tests."""


@pytest.fixture(scope="session")
def model_bytes() -> bytes:
    """A small Teachable-Machine-like float model with 3 classes."""
    return build_teachable_machine_like(num_classes=3, size=64)


def make_zip(
    path: Path,
    model: bytes | None,
    labels: list[str] | None,
    model_name: str = "model_unquant.tflite",
) -> Path:
    """Write a Teachable Machine style export archive."""
    with zipfile.ZipFile(path, "w") as archive:
        if model is not None:
            archive.writestr(model_name, model)
        if labels is not None:
            archive.writestr(
                "labels.txt", "".join(f"{i} {name}\n" for i, name in enumerate(labels))
            )
    return path


@pytest.fixture
def model_zip(tmp_path: Path, model_bytes: bytes) -> Path:
    """A valid export archive."""
    return make_zip(tmp_path / "converted_tflite.zip", model_bytes, LABELS)


@contextmanager
def mock_upload(path: Path) -> Generator[None]:
    """Make ``process_uploaded_file`` return ``path`` for any file id."""

    @contextmanager
    def _process(hass, file_id):
        yield path

    with patch(
        "custom_components.snapshot_classifier.config_flow.process_uploaded_file",
        _process,
    ):
        yield


def jpeg(color: tuple[int, int, int] = (120, 60, 30), size=(656, 368)) -> bytes:
    """Return a JPEG image."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def camera_image() -> Generator[list[bytes]]:
    """Patch the camera so it returns the last item of the yielded list."""
    images = [jpeg()]

    async def _get_image(hass, entity_id, timeout=10, width=None, height=None):
        return CameraImage("image/jpeg", images[-1])

    with patch(
        "custom_components.snapshot_classifier.coordinator.camera.async_get_image",
        side_effect=_get_image,
    ):
        yield images
