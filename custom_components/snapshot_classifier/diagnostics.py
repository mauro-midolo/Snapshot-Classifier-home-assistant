"""Diagnostics support for Snapshot Classifier."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import SnapshotClassifierConfigEntry


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: SnapshotClassifierConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = entry.runtime_data
    data = coordinator.data
    return {
        "data": dict(entry.data),
        "options": dict(entry.options),
        "model": {
            "labels": coordinator.labels,
            "input_size": list(coordinator.classifier.input_size),
            "sha256": coordinator.classifier.sha256,
        },
        "last_update_success": coordinator.last_update_success,
        "last_result": None
        if data is None
        else {
            "state": data.state,
            "probabilities": data.probabilities,
            "snapshot_time": data.snapshot_time.isoformat(),
            "analyzed_at": data.analyzed_at.isoformat(),
            "duration": data.duration,
        },
    }
