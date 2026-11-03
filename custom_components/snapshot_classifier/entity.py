"""Base entity for Snapshot Classifier."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SnapshotClassifierCoordinator


class SnapshotClassifierEntity(CoordinatorEntity[SnapshotClassifierCoordinator]):
    """An entity belonging to one classifier (config entry)."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SnapshotClassifierCoordinator, key: str) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Snapshot Classifier",
            model="Teachable Machine image model",
            entry_type=DeviceEntryType.SERVICE,
        )
