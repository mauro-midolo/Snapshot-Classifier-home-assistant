"""Services of the Snapshot Classifier integration."""

from __future__ import annotations

from pathlib import Path

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .classifier import CropBox, export_crops
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_DESTINATION,
    ATTR_HEIGHT,
    ATTR_LEFT,
    ATTR_OVERWRITE,
    ATTR_SOURCE,
    ATTR_TOP,
    ATTR_WIDTH,
    CONF_CROP,
    CONF_CROP_HEIGHT,
    CONF_CROP_WIDTH,
    CONF_CROP_X,
    CONF_CROP_Y,
    DOMAIN,
    SERVICE_EXPORT_CROPS,
)

_FRACTION = vol.All(vol.Coerce(float), vol.Range(min=0, max=1))

EXPORT_CROPS_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_SOURCE): cv.string,
        vol.Required(ATTR_DESTINATION): cv.string,
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Optional(ATTR_LEFT): _FRACTION,
        vol.Optional(ATTR_TOP): _FRACTION,
        vol.Optional(ATTR_WIDTH): _FRACTION,
        vol.Optional(ATTR_HEIGHT): _FRACTION,
        vol.Optional(ATTR_OVERWRITE, default=False): cv.boolean,
    }
)


def _crop_from_call(hass: HomeAssistant, call: ServiceCall) -> CropBox:
    """Return the crop of the selected classifier, overridden by explicit fields."""
    base: dict[str, float] = {}
    if entry_id := call.data.get(ATTR_CONFIG_ENTRY_ID):
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"entry_id": entry_id},
            )
        base = dict(entry.options.get(CONF_CROP) or {})
    fields = {
        ATTR_LEFT: CONF_CROP_X,
        ATTR_TOP: CONF_CROP_Y,
        ATTR_WIDTH: CONF_CROP_WIDTH,
        ATTR_HEIGHT: CONF_CROP_HEIGHT,
    }
    for field, key in fields.items():
        if field in call.data:
            base[key] = call.data[field]
    crop = CropBox.from_mapping(base)
    if not crop.is_valid:
        raise ServiceValidationError(translation_domain=DOMAIN, translation_key="invalid_crop")
    return crop


def _checked_dir(hass: HomeAssistant, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not hass.config.is_allowed_path(str(path)):
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="path_not_allowed",
            translation_placeholders={"path": value},
        )
    return path


async def _async_export_crops(call: ServiceCall) -> ServiceResponse:
    hass = call.hass
    crop = _crop_from_call(hass, call)
    source = _checked_dir(hass, call.data[ATTR_SOURCE])
    destination = _checked_dir(hass, call.data[ATTR_DESTINATION])
    if not await hass.async_add_executor_job(source.is_dir):
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="source_not_found",
            translation_placeholders={"path": str(source)},
        )
    if source.resolve() == destination.resolve():
        raise ServiceValidationError(translation_domain=DOMAIN, translation_key="same_directory")
    result = await hass.async_add_executor_job(
        export_crops, source, destination, crop, call.data[ATTR_OVERWRITE]
    )
    return {
        "exported": result.exported,
        "skipped": result.skipped,
        "failed": result.failed,
        "crop": {
            ATTR_LEFT: crop.x,
            ATTR_TOP: crop.y,
            ATTR_WIDTH: crop.width,
            ATTR_HEIGHT: crop.height,
        },
    }


def async_setup_services(hass: HomeAssistant) -> None:
    """Register the integration services."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_CROPS,
        _async_export_crops,
        schema=EXPORT_CROPS_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
