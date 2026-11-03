"""Event entity fired when a snapshot cannot be classified confidently."""

from __future__ import annotations

from homeassistant.components.event import EventEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import EVENT_UNCERTAIN
from .coordinator import (
    ClassificationResult,
    SnapshotClassifierConfigEntry,
    SnapshotClassifierCoordinator,
)
from .entity import SnapshotClassifierEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SnapshotClassifierConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the event entity."""
    async_add_entities([UncertainEvent(entry.runtime_data)])


class UncertainEvent(SnapshotClassifierEntity, EventEntity):
    """Fires 'uncertain' when the best class is under the threshold.

    Useful to find the pictures worth adding to the training set.
    """

    _attr_translation_key = "uncertain_result"

    def __init__(self, coordinator: SnapshotClassifierCoordinator) -> None:
        """Initialize the entity."""
        super().__init__(coordinator, "uncertain_result")
        self._attr_event_types = [EVENT_UNCERTAIN]
        self._last: ClassificationResult | None = coordinator.data

    @property
    def available(self) -> bool:
        """Keep the last event visible even when an update fails."""
        return True

    @callback
    def _handle_coordinator_update(self) -> None:
        data = self.coordinator.data
        if data is None or data is self._last:
            return
        self._last = data
        if not data.uncertain:
            return
        attributes = {
            "best_class": data.best_class,
            "confidence": round(data.confidence * 100, 2),
            "probabilities": {label: round(p * 100, 2) for label, p in data.probabilities.items()},
            "snapshot_time": data.snapshot_time.isoformat(),
        }
        if data.saved_image:
            attributes["image_path"] = data.saved_image
        self._trigger_event(EVENT_UNCERTAIN, attributes)
        self.async_write_ha_state()
