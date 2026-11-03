"""Coordinator fetching camera snapshots and classifying them."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
import time
from typing import Any

from homeassistant.components import camera
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util, slugify

from .classifier import Classifier, CropBox
from .const import (
    CONF_CAMERA,
    CONF_CROP,
    CONF_EMPTY_CLASS,
    CONF_SCAN_INTERVAL,
    CONF_TARGET_CLASS,
    CONF_THRESHOLD,
    CONF_UNCERTAIN_DIR,
    CONF_UPDATE_MODE,
    CONF_WATCH_ATTRIBUTE,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_THRESHOLD,
    DEFAULT_WATCH_ATTRIBUTE,
    DOMAIN,
    STATE_UNCERTAIN,
    UPDATE_MODE_ATTRIBUTE,
    UPDATE_MODE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

IMAGE_TIMEOUT = 30


@dataclass(frozen=True, slots=True)
class Settings:
    """Options of a config entry, with defaults applied."""

    camera_entity_id: str
    threshold: float
    update_mode: str
    watch_attribute: str
    scan_interval: int
    crop: CropBox
    empty_class: str | None
    target_class: str | None
    uncertain_dir: str | None

    @classmethod
    def from_entry(cls, entry: ConfigEntry) -> Settings:
        """Read the settings of a config entry."""
        options = entry.options
        return cls(
            camera_entity_id=entry.data[CONF_CAMERA],
            threshold=float(options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
            update_mode=options.get(CONF_UPDATE_MODE, UPDATE_MODE_ATTRIBUTE),
            watch_attribute=options.get(CONF_WATCH_ATTRIBUTE) or DEFAULT_WATCH_ATTRIBUTE,
            scan_interval=int(options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)),
            crop=CropBox.from_mapping(options.get(CONF_CROP)),
            empty_class=options.get(CONF_EMPTY_CLASS) or None,
            target_class=options.get(CONF_TARGET_CLASS) or None,
            uncertain_dir=options.get(CONF_UNCERTAIN_DIR) or None,
        )


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Outcome of one analysis."""

    probabilities: dict[str, float]
    best_class: str
    confidence: float
    uncertain: bool
    snapshot_time: datetime
    analyzed_at: datetime
    duration: float
    saved_image: str | None = None

    @property
    def state(self) -> str:
        """Return the best class, or 'uncertain'."""
        return STATE_UNCERTAIN if self.uncertain else self.best_class

    def probability(self, label: str | None) -> float | None:
        """Return the probability of a label."""
        if label is None:
            return None
        return self.probabilities.get(label)


def parse_snapshot_time(value: Any) -> datetime | None:
    """Interpret a camera attribute (epoch seconds/ms or ISO string) as a time."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return dt_util.as_utc(value) if value.tzinfo else value.replace(tzinfo=dt_util.UTC)
    try:
        number = float(value)
    except (TypeError, ValueError):
        parsed = dt_util.parse_datetime(str(value))
        if parsed is None:
            return None
        return dt_util.as_utc(parsed) if parsed.tzinfo else parsed.replace(tzinfo=dt_util.UTC)
    if number > 1e12:  # milliseconds
        number /= 1000
    try:
        return dt_util.utc_from_timestamp(number)
    except (OverflowError, OSError, ValueError):
        return None


type SnapshotClassifierConfigEntry = ConfigEntry[SnapshotClassifierCoordinator]


class SnapshotClassifierCoordinator(DataUpdateCoordinator[ClassificationResult | None]):
    """Fetch the camera image and classify it."""

    config_entry: SnapshotClassifierConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        entry: SnapshotClassifierConfigEntry,
        classifier: Classifier,
    ) -> None:
        """Initialize the coordinator."""
        self.settings = Settings.from_entry(entry)
        self.classifier = classifier
        interval = None
        if self.settings.update_mode == UPDATE_MODE_INTERVAL:
            interval = timedelta(seconds=self.settings.scan_interval)
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title}",
            update_interval=interval,
            always_update=False,
        )
        self.data = None
        self._lock = asyncio.Lock()
        self._last_digest: str | None = None
        self._force = False

    @property
    def labels(self) -> list[str]:
        """Return the model classes."""
        return self.classifier.labels

    async def async_analyze_now(self) -> None:
        """Analyze the current snapshot even if it did not change."""
        self._force = True
        await self.async_refresh()

    def _snapshot_time(self) -> datetime | None:
        if self.settings.update_mode != UPDATE_MODE_ATTRIBUTE:
            return None
        state = self.hass.states.get(self.settings.camera_entity_id)
        if state is None:
            return None
        return parse_snapshot_time(state.attributes.get(self.settings.watch_attribute))

    async def _async_update_data(self) -> ClassificationResult | None:
        async with self._lock:
            force, self._force = self._force, False
            try:
                image = await camera.async_get_image(
                    self.hass, self.settings.camera_entity_id, timeout=IMAGE_TIMEOUT
                )
            except HomeAssistantError as err:
                raise UpdateFailed(
                    f"Cannot get an image from {self.settings.camera_entity_id}: {err}"
                ) from err

            digest = hashlib.sha256(image.content).hexdigest()
            if digest == self._last_digest and not force and self.data is not None:
                _LOGGER.debug("%s: snapshot unchanged, skipping", self.name)
                return self.data

            started = time.perf_counter()
            try:
                probabilities = await self.hass.async_add_executor_job(
                    self.classifier.classify, image.content, self.settings.crop
                )
            except ValueError as err:
                raise UpdateFailed(str(err)) from err
            duration = time.perf_counter() - started
            self._last_digest = digest

            best_class = max(probabilities, key=probabilities.__getitem__)
            confidence = probabilities[best_class]
            uncertain = confidence * 100 < self.settings.threshold
            now = dt_util.utcnow()

            saved_image = None
            if uncertain and self.settings.uncertain_dir:
                saved_image = await self._async_save_uncertain(
                    image.content, image.content_type, now
                )

            _LOGGER.debug(
                "%s: %s (%.1f%%) in %.3fs: %s",
                self.name,
                best_class,
                confidence * 100,
                duration,
                probabilities,
            )
            return ClassificationResult(
                probabilities=probabilities,
                best_class=best_class,
                confidence=confidence,
                uncertain=uncertain,
                snapshot_time=self._snapshot_time() or now,
                analyzed_at=now,
                duration=duration,
                saved_image=saved_image,
            )

    async def _async_save_uncertain(
        self, content: bytes, content_type: str, now: datetime
    ) -> str | None:
        """Store an uncertain snapshot so it can be added to the training set."""
        directory = Path(self.settings.uncertain_dir or "")
        if not self.hass.config.is_allowed_path(str(directory)):
            _LOGGER.warning(
                "%s is not an allowed directory (allowlist_external_dirs), "
                "uncertain snapshot not saved",
                directory,
            )
            return None
        extension = {"image/png": "png", "image/webp": "webp"}.get(content_type, "jpg")
        local = dt_util.as_local(now)
        path = directory / (f"{slugify(self.config_entry.title)}_{local:%Y%m%d_%H%M%S}.{extension}")

        def _write() -> None:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        try:
            await self.hass.async_add_executor_job(_write)
        except OSError as err:
            _LOGGER.warning("Cannot save uncertain snapshot to %s: %s", path, err)
            return None
        return str(path)
