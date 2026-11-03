"""Image classification with a Teachable Machine model.

Pure Python/numpy/Pillow code with no Home Assistant dependency. Everything
here is blocking and must be run in the executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .const import (
    CONF_CROP_HEIGHT,
    CONF_CROP_WIDTH,
    CONF_CROP_X,
    CONF_CROP_Y,
    IMAGE_EXTENSIONS,
    LABELS_FILENAME,
    MODEL_FILENAME,
)
from .tflite_numpy import InvalidModelError, TFLiteModel, UnsupportedModelError

_LOGGER = logging.getLogger(__name__)

# Teachable Machine names the float model like this; the quantized export
# uses "model.tflite".
_PREFERRED_MODEL_NAME = "model_unquant.tflite"
_MAX_MEMBER_SIZE = 256 * 1024 * 1024
_LABEL_RE = re.compile(r"^\s*(\d+)\s+(.*?)\s*$")


class ModelArchiveError(Exception):
    """The uploaded model archive cannot be used.

    ``reason`` is a translation key used by the config flow.
    """

    def __init__(self, reason: str, **placeholders: str) -> None:
        """Initialize the error."""
        super().__init__(reason, placeholders)
        self.reason = reason
        self.placeholders = placeholders


@dataclass(frozen=True, slots=True)
class CropBox:
    """A crop rectangle in normalized (0-1) image coordinates."""

    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> CropBox:
        """Build a crop box from options."""
        data = data or {}
        return cls(
            x=float(data.get(CONF_CROP_X, 0.0)),
            y=float(data.get(CONF_CROP_Y, 0.0)),
            width=float(data.get(CONF_CROP_WIDTH, 1.0)),
            height=float(data.get(CONF_CROP_HEIGHT, 1.0)),
        )

    def as_dict(self) -> dict[str, float]:
        """Return the options representation."""
        return {
            CONF_CROP_X: self.x,
            CONF_CROP_Y: self.y,
            CONF_CROP_WIDTH: self.width,
            CONF_CROP_HEIGHT: self.height,
        }

    @property
    def is_valid(self) -> bool:
        """Return True if the box lies inside the image and is not empty."""
        eps = 1e-6
        return (
            0 <= self.x < 1
            and 0 <= self.y < 1
            and self.width > 0
            and self.height > 0
            and self.x + self.width <= 1 + eps
            and self.y + self.height <= 1 + eps
        )

    @property
    def is_full_image(self) -> bool:
        """Return True if the box keeps the whole image."""
        return (self.x, self.y, self.width, self.height) == (0.0, 0.0, 1.0, 1.0)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the (left, top, right, bottom) pixel box for an image size."""
        left = min(max(round(self.x * width), 0), width - 1)
        top = min(max(round(self.y * height), 0), height - 1)
        right = min(max(round((self.x + self.width) * width), left + 1), width)
        bottom = min(max(round((self.y + self.height) * height), top + 1), height)
        return left, top, right, bottom


def parse_labels(text: str) -> list[str]:
    """Parse a Teachable Machine labels.txt ("0 Class name" per line)."""
    labels: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _LABEL_RE.match(line)
        labels.append(match.group(2) if match else line.strip())
    return labels


def open_image(data: bytes) -> Image.Image:
    """Decode an image, honour EXIF orientation and convert to RGB."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image) or image
    return image.convert("RGB")


def crop_image(image: Image.Image, crop: CropBox) -> Image.Image:
    """Apply the normalized crop box."""
    if crop.is_full_image:
        return image
    return image.crop(crop.pixels(*image.size))


def preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Replicate Teachable Machine's preprocessing.

    Centre crop to the model aspect ratio (a square for Teachable Machine),
    bilinear resize and scale pixels to [-1, 1]. Returns [1, H, W, 3] float32.
    """
    src_w, src_h = image.size
    target_ratio = width / height
    if src_w / src_h > target_ratio:
        new_w = max(round(src_h * target_ratio), 1)
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        new_h = max(round(src_w / target_ratio), 1)
        top = (src_h - new_h) // 2
        box = (0, top, src_w, top + new_h)
    if box != (0, 0, src_w, src_h):
        image = image.crop(box)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return (array / 127.5 - 1.0)[np.newaxis, ...]


@dataclass(slots=True)
class Classifier:
    """A loaded model plus its labels."""

    model_data: bytes
    labels: list[str]
    _model: TFLiteModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Parse and validate the model. Raises ModelArchiveError."""
        if not self.labels:
            raise ModelArchiveError("invalid_labels")
        if len(set(self.labels)) != len(self.labels):
            raise ModelArchiveError("invalid_labels")
        try:
            self._model = TFLiteModel(self.model_data)
        except UnsupportedModelError as err:
            if err.quantized:
                raise ModelArchiveError("unsupported_quantized") from err
            raise ModelArchiveError("unsupported_model", details=str(err)) from err
        except InvalidModelError as err:
            raise ModelArchiveError("invalid_model") from err

        shape = self._model.input_shape
        if len(shape) != 4 or shape[3] != 3 or shape[1] <= 0 or shape[2] <= 0:
            raise ModelArchiveError(
                "unsupported_model",
                details=f"expected an RGB image input [1, H, W, 3], got {list(shape)}",
            )
        # One dry run also catches problems that only show up at run time.
        try:
            output = self._model.predict(np.zeros((1, *shape[1:]), dtype=np.float32))
        except Exception as err:  # any failure means the model is unusable
            raise ModelArchiveError("unsupported_model", details=str(err)) from err
        if output.size != len(self.labels):
            raise ModelArchiveError(
                "label_mismatch",
                outputs=str(output.size),
                labels=str(len(self.labels)),
            )

    @property
    def input_size(self) -> tuple[int, int]:
        """Return the model input (width, height)."""
        return self._model.input_shape[2], self._model.input_shape[1]

    @property
    def sha256(self) -> str:
        """Return the model fingerprint."""
        return hashlib.sha256(self.model_data).hexdigest()

    def classify_image(self, image: Image.Image, crop: CropBox) -> dict[str, float]:
        """Return the probability of every label for a decoded image."""
        data = preprocess(crop_image(image, crop), *self.input_size)
        output = self._model.predict(data).reshape(-1).astype(np.float64)
        total = float(output.sum())
        if output.min() < 0 or not 0.99 <= total <= 1.01:
            # Not a softmax output: normalize it.
            output = np.exp(output - output.max())
            total = float(output.sum())
        output /= total
        return {label: float(p) for label, p in zip(self.labels, output, strict=True)}

    def classify(self, image_data: bytes, crop: CropBox) -> dict[str, float]:
        """Return the probability of every label for encoded image bytes."""
        try:
            image = open_image(image_data)
        except (UnidentifiedImageError, OSError) as err:
            raise ValueError(f"Cannot decode camera image: {err}") from err
        return self.classify_image(image, crop)


# --------------------------------------------------------------------------- #
# Model archive and storage
# --------------------------------------------------------------------------- #


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > _MAX_MEMBER_SIZE:
        raise ModelArchiveError("invalid_archive")
    return archive.read(info)


def read_model_archive(path: str | os.PathLike[str]) -> Classifier:
    """Read and validate a Teachable Machine TensorFlow Lite export (.zip)."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = {
                Path(info.filename).name: info
                for info in archive.infolist()
                if not info.is_dir() and not Path(info.filename).name.startswith(".")
            }
            models = [name for name in members if name.lower().endswith(".tflite")]
            if _PREFERRED_MODEL_NAME in members:
                model_name = _PREFERRED_MODEL_NAME
            elif len(models) == 1:
                model_name = models[0]
            else:
                raise ModelArchiveError("missing_model")
            if LABELS_FILENAME not in members:
                raise ModelArchiveError("missing_labels")
            model_data = _read_member(archive, members[model_name])
            labels_data = _read_member(archive, members[LABELS_FILENAME])
    except (zipfile.BadZipFile, OSError, EOFError) as err:
        raise ModelArchiveError("invalid_archive") from err

    try:
        labels = parse_labels(labels_data.decode("utf-8-sig"))
    except UnicodeDecodeError as err:
        raise ModelArchiveError("invalid_labels") from err
    return Classifier(model_data, labels)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_model(directory: Path, classifier: Classifier) -> None:
    """Store model and labels in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    labels = "".join(f"{i} {label}\n" for i, label in enumerate(classifier.labels))
    _atomic_write(directory / LABELS_FILENAME, labels.encode())
    _atomic_write(directory / MODEL_FILENAME, classifier.model_data)


def load_model(directory: Path) -> Classifier:
    """Load a model saved with :func:`save_model`."""
    model_data = (directory / MODEL_FILENAME).read_bytes()
    labels = parse_labels((directory / LABELS_FILENAME).read_text(encoding="utf-8"))
    return Classifier(model_data, labels)


# --------------------------------------------------------------------------- #
# Training data helper
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ExportResult:
    """Outcome of :func:`export_crops`."""

    exported: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)


def export_crops(source: Path, destination: Path, crop: CropBox, overwrite: bool) -> ExportResult:
    """Crop every image of ``source`` into ``destination`` (same file names)."""
    result = ExportResult()
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = destination / path.name
        if target.exists() and not overwrite:
            result.skipped += 1
            continue
        try:
            with Image.open(path) as original:
                fmt = original.format
                image = ImageOps.exif_transpose(original) or original
                cropped = crop_image(image, crop)
                if fmt == "JPEG" and cropped.mode not in ("RGB", "L"):
                    cropped = cropped.convert("RGB")
                save_args: dict[str, Any] = {"quality": 95} if fmt == "JPEG" else {}
                cropped.save(target, format=fmt, **save_args)
        except (UnidentifiedImageError, OSError, ValueError) as err:
            _LOGGER.warning("Cannot crop %s: %s", path, err)
            result.failed.append(path.name)
            continue
        result.exported += 1
    return result
