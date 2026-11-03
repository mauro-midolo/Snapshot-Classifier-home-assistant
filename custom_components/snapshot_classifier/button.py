"""Button to analyze the current snapshot on demand."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import SnapshotClassifierConfigEntry, SnapshotClassifierCoordinator
from .entity import SnapshotClassifierEntity

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SnapshotClassifierConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the button."""
    async_add_entities([AnalyzeButton(entry.runtime_data)])


class AnalyzeButton(SnapshotClassifierEntity, ButtonEntity):
    """Analyze the current camera snapshot now."""

    _attr_translation_key = "analyze"

    def __init__(self, coordinator: SnapshotClassifierCoordinator) -> None:
        """Initialize the button."""
        super().__init__(coordinator, "analyze")

    @property
    def available(self) -> bool:
        """The button is usable even after a failed analysis."""
        return True

    async def async_press(self) -> None:
        """Run an analysis."""
        await self.coordinator.async_analyze_now()
