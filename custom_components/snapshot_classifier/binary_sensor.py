"""Binary sensors derived from the classification."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN, OCCUPANCY_PROBABILITY
from .coordinator import SnapshotClassifierConfigEntry, SnapshotClassifierCoordinator
from .entity import SnapshotClassifierEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SnapshotClassifierConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the binary sensors selected in the configuration."""
    coordinator = entry.runtime_data
    settings = coordinator.settings
    entities: list[BinarySensorEntity] = []
    if settings.empty_class in coordinator.labels:
        entities.append(OccupiedBinarySensor(coordinator, settings.empty_class))
    if settings.target_class in coordinator.labels:
        entities.append(ClassPresentBinarySensor(coordinator, settings.target_class))
    async_add_entities(entities)

    # Drop the entities of classes that are no longer selected in the options.
    registry = er.async_get(hass)
    wanted = {entity.unique_id for entity in entities}
    for key in ("occupied", "class_present"):
        unique_id = f"{entry.entry_id}_{key}"
        if unique_id in wanted:
            continue
        if entity_id := registry.async_get_entity_id("binary_sensor", DOMAIN, unique_id):
            registry.async_remove(entity_id)


class _ProbabilityBinarySensor(SnapshotClassifierEntity, BinarySensorEntity):
    """Binary sensor based on the probability of one class."""

    def __init__(self, coordinator: SnapshotClassifierCoordinator, key: str, label: str) -> None:
        super().__init__(coordinator, key)
        self._label = label

    @property
    def _probability(self) -> float | None:
        data = self.coordinator.data
        return data.probability(self._label) if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the class and its probability."""
        probability = self._probability
        if probability is None:
            return None
        return {"class": self._label, "probability": round(probability * 100, 2)}


class OccupiedBinarySensor(_ProbabilityBinarySensor):
    """On when the 'empty' class is less likely than 50%."""

    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_translation_key = "occupied"

    def __init__(self, coordinator: SnapshotClassifierCoordinator, label: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "occupied", label)

    @property
    def is_on(self) -> bool | None:
        """Return True if the place is occupied."""
        probability = self._probability
        return None if probability is None else probability < OCCUPANCY_PROBABILITY


class ClassPresentBinarySensor(_ProbabilityBinarySensor):
    """On when the chosen class (e.g. 'my car') is more likely than 50%."""

    _attr_translation_key = "class_present"

    def __init__(self, coordinator: SnapshotClassifierCoordinator, label: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "class_present", label)
        self._attr_translation_placeholders = {"class_name": label}

    @property
    def is_on(self) -> bool | None:
        """Return True if the class is detected."""
        probability = self._probability
        return None if probability is None else probability >= OCCUPANCY_PROBABILITY
