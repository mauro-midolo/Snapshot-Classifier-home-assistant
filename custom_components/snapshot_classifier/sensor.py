"""Sensors with the classification result."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import STATE_UNCERTAIN
from .coordinator import SnapshotClassifierConfigEntry, SnapshotClassifierCoordinator
from .entity import SnapshotClassifierEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SnapshotClassifierConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the sensors."""
    coordinator = entry.runtime_data
    async_add_entities(
        [
            ClassificationSensor(coordinator),
            ConfidenceSensor(coordinator),
        ]
    )


class ClassificationSensor(SnapshotClassifierEntity, SensorEntity):
    """The most likely class, or 'uncertain' under the threshold."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_translation_key = "classification"

    def __init__(self, coordinator: SnapshotClassifierCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "classification")
        self._attr_options = [*coordinator.labels, STATE_UNCERTAIN]

    @property
    def native_value(self) -> str | None:
        """Return the classification."""
        data = self.coordinator.data
        return data.state if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the probabilities and timing information."""
        data = self.coordinator.data
        if data is None:
            return None
        return {
            "best_class": data.best_class,
            "probabilities": {label: round(p * 100, 2) for label, p in data.probabilities.items()},
            "threshold": self.coordinator.settings.threshold,
            "snapshot_time": data.snapshot_time.isoformat(),
            "analyzed_at": data.analyzed_at.isoformat(),
            "inference_seconds": round(data.duration, 3),
        }


class ConfidenceSensor(SnapshotClassifierEntity, SensorEntity):
    """Probability of the most likely class."""

    _attr_translation_key = "confidence"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: SnapshotClassifierCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "confidence")

    @property
    def native_value(self) -> float | None:
        """Return the confidence in percent."""
        data = self.coordinator.data
        return round(data.confidence * 100, 2) if data else None
