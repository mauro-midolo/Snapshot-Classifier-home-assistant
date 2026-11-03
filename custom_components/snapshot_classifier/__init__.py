"""Classify Home Assistant camera snapshots with a Teachable Machine model."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType

from .classifier import ModelArchiveError, load_model
from .const import CONF_STORAGE_ID, DOMAIN, UPDATE_MODE_ATTRIBUTE, UPDATE_MODE_MANUAL
from .coordinator import SnapshotClassifierConfigEntry, SnapshotClassifierCoordinator
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.EVENT, Platform.SENSOR]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def model_directory(hass: HomeAssistant, storage_id: str) -> Path:
    """Return where the model of a config entry is stored."""
    return Path(hass.config.path(".storage", DOMAIN, storage_id))


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration (services)."""
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: SnapshotClassifierConfigEntry) -> bool:
    """Set up a classifier from a config entry."""
    directory = model_directory(hass, entry.data[CONF_STORAGE_ID])
    try:
        classifier = await hass.async_add_executor_job(load_model, directory)
    except FileNotFoundError as err:
        raise ConfigEntryError(
            translation_domain=DOMAIN,
            translation_key="model_missing",
            translation_placeholders={"path": str(directory)},
        ) from err
    except ModelArchiveError as err:
        raise ConfigEntryError(
            translation_domain=DOMAIN,
            translation_key="model_invalid",
            translation_placeholders={"error": err.reason},
        ) from err

    coordinator = SnapshotClassifierCoordinator(hass, entry, classifier)
    entry.runtime_data = coordinator
    settings = coordinator.settings

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    if settings.update_mode == UPDATE_MODE_ATTRIBUTE:

        @callback
        def _camera_changed(event: Event[EventStateChangedData]) -> None:
            old_state = event.data["old_state"]
            new_state = event.data["new_state"]
            if new_state is None:
                return
            new_value = new_state.attributes.get(settings.watch_attribute)
            old_value = old_state.attributes.get(settings.watch_attribute) if old_state else None
            if new_value is None or new_value == old_value:
                return
            entry.async_create_task(
                hass,
                coordinator.async_request_refresh(),
                f"{DOMAIN} {entry.title} snapshot changed",
            )

        entry.async_on_unload(
            async_track_state_change_event(hass, settings.camera_entity_id, _camera_changed)
        )

    if settings.update_mode != UPDATE_MODE_MANUAL:

        @callback
        def _initial_analysis(_hass: HomeAssistant) -> None:
            entry.async_create_task(
                hass, coordinator.async_refresh(), f"{DOMAIN} {entry.title} initial analysis"
            )

        entry.async_on_unload(async_at_started(hass, _initial_analysis))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options (or the model) change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SnapshotClassifierConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the stored model when the entry is removed."""
    storage_id = entry.data.get(CONF_STORAGE_ID)
    if not storage_id:
        return
    directory = model_directory(hass, storage_id)
    await hass.async_add_executor_job(shutil.rmtree, directory, True)
