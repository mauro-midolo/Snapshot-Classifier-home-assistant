"""Tests for the config and options flow."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.snapshot_classifier import model_directory
from custom_components.snapshot_classifier.classifier import Classifier, save_model
from custom_components.snapshot_classifier.const import (
    CONF_CAMERA,
    CONF_CROP,
    CONF_EMPTY_CLASS,
    CONF_MODEL_FILE,
    CONF_MODEL_SHA256,
    CONF_SCAN_INTERVAL,
    CONF_STORAGE_ID,
    CONF_TARGET_CLASS,
    CONF_THRESHOLD,
    CONF_UNCERTAIN_DIR,
    CONF_UPDATE_MODE,
    CONF_WATCH_ATTRIBUTE,
    DOMAIN,
    LABELS_FILENAME,
    MODEL_FILENAME,
    UPDATE_MODE_ATTRIBUTE,
    UPDATE_MODE_INTERVAL,
)
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from .conftest import LABELS, make_zip, mock_upload
from .tflite_builder import (
    build_quantized_model,
    build_teachable_machine_like,
    build_unsupported_op_model,
)

USER_INPUT = {
    CONF_NAME: "Parcheggio",
    CONF_CAMERA: "camera.ingresso",
    CONF_MODEL_FILE: "6f1c2a52-5c1e-4a43-9a8e-0d6f1b7a1a01",
}


@pytest.mark.usefixtures("camera_image")
async def test_full_flow(hass: HomeAssistant, model_zip: Path) -> None:
    """Upload a model, choose the classes, get an entry with the model stored."""
    hass.states.async_set("camera.ingresso", "idle", {"timestamp": 1700000000})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with mock_upload(model_zip):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], USER_INPUT)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "classes"
    assert result["description_placeholders"] == {"classes": "Mine, Other, Empty"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_EMPTY_CLASS: "Empty", CONF_TARGET_CLASS: "Mine"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Parcheggio"
    entry = result["result"]
    assert entry.data[CONF_CAMERA] == "camera.ingresso"
    assert entry.options[CONF_EMPTY_CLASS] == "Empty"
    assert entry.options[CONF_TARGET_CLASS] == "Mine"
    assert entry.options[CONF_UPDATE_MODE] == UPDATE_MODE_ATTRIBUTE
    assert entry.options[CONF_THRESHOLD] == 70

    directory = model_directory(hass, entry.data[CONF_STORAGE_ID])
    assert (directory / MODEL_FILENAME).is_file()
    assert (directory / LABELS_FILENAME).read_text() == "0 Mine\n1 Other\n2 Empty\n"
    assert hass.states.get("sensor.parcheggio_status") is not None


@pytest.mark.usefixtures("camera_image")
async def test_flow_without_timestamp_uses_interval(hass: HomeAssistant, model_zip: Path) -> None:
    """Cameras without a timestamp attribute default to interval updates."""
    hass.states.async_set("camera.ingresso", "idle")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with mock_upload(model_zip):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], USER_INPUT)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"][CONF_UPDATE_MODE] == UPDATE_MODE_INTERVAL
    assert CONF_EMPTY_CLASS not in result["options"]


@pytest.mark.parametrize(
    ("archive", "error"),
    [
        ("quantized", "unsupported_quantized"),
        ("unsupported", "unsupported_model"),
        ("no_labels", "missing_labels"),
        ("no_model", "missing_model"),
        ("mismatch", "label_mismatch"),
        ("not_zip", "invalid_archive"),
        ("garbage_model", "invalid_model"),
    ],
)
async def test_flow_rejects_bad_archives(
    hass: HomeAssistant, tmp_path: Path, model_bytes: bytes, archive: str, error: str
) -> None:
    """Every kind of unusable upload is reported on the file field."""
    path = tmp_path / "upload.zip"
    if archive == "quantized":
        make_zip(path, build_quantized_model(), ["a", "b"], "model.tflite")
    elif archive == "unsupported":
        make_zip(path, build_unsupported_op_model(), ["a"])
    elif archive == "no_labels":
        make_zip(path, model_bytes, None)
    elif archive == "no_model":
        make_zip(path, None, LABELS)
    elif archive == "mismatch":
        make_zip(path, model_bytes, ["a", "b"])
    elif archive == "garbage_model":
        make_zip(path, b"0123456789abcdef", LABELS)
    else:
        path.write_bytes(b"not a zip")

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    with mock_upload(path):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], USER_INPUT)
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_MODEL_FILE: error}


async def _setup_entry(hass: HomeAssistant, model_bytes: bytes) -> MockConfigEntry:
    hass.states.async_set("camera.ingresso", "idle", {"timestamp": 1})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Parcheggio",
        data={CONF_CAMERA: "camera.ingresso", CONF_STORAGE_ID: "abc"},
        options={
            CONF_THRESHOLD: 70,
            CONF_UPDATE_MODE: UPDATE_MODE_ATTRIBUTE,
            CONF_EMPTY_CLASS: "Empty",
        },
    )
    entry.add_to_hass(hass)
    await hass.async_add_executor_job(
        save_model, model_directory(hass, "abc"), Classifier(model_bytes, LABELS)
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _options_input(**overrides: object) -> dict:
    data = {
        CONF_CAMERA: "camera.ingresso",
        CONF_THRESHOLD: 80,
        CONF_UPDATE_MODE: UPDATE_MODE_INTERVAL,
        CONF_WATCH_ATTRIBUTE: "timestamp",
        CONF_SCAN_INTERVAL: 30,
        CONF_EMPTY_CLASS: "Empty",
        CONF_CROP: {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5},
    }
    data.update(overrides)
    return data


@pytest.mark.usefixtures("camera_image")
async def test_options_flow(hass: HomeAssistant, model_bytes: bytes) -> None:
    """Change the settings without a new model."""
    entry = await _setup_entry(hass, model_bytes)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["description_placeholders"]["classes"] == "Mine, Other, Empty"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        _options_input(**{CONF_CROP: {"x": 0.6, "y": 0, "width": 0.5, "height": 1}}),
    )
    assert result["errors"] == {CONF_CROP: "invalid_crop"}

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _options_input(**{CONF_UNCERTAIN_DIR: "/not/allowed"})
    )
    assert result["errors"] == {CONF_UNCERTAIN_DIR: "path_not_allowed"}

    result = await hass.config_entries.options.async_configure(result["flow_id"], _options_input())
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_THRESHOLD] == 80
    assert entry.options[CONF_UPDATE_MODE] == UPDATE_MODE_INTERVAL
    assert entry.options[CONF_CROP] == {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5}
    assert CONF_TARGET_CLASS not in entry.options
    assert entry.runtime_data.settings.threshold == 80

    # Clearing the "empty" class removes the Occupied sensor
    assert hass.states.get("binary_sensor.parcheggio_occupied") is not None
    result = await hass.config_entries.options.async_init(entry.entry_id)
    user_input = _options_input()
    del user_input[CONF_EMPTY_CLASS]
    await hass.config_entries.options.async_configure(result["flow_id"], user_input)
    await hass.async_block_till_done()
    assert CONF_EMPTY_CLASS not in entry.options
    assert hass.states.get("binary_sensor.parcheggio_occupied") is None


@pytest.mark.usefixtures("camera_image")
async def test_options_flow_replaces_model(
    hass: HomeAssistant, tmp_path: Path, model_bytes: bytes
) -> None:
    """A retrained model with new classes asks for the classes again."""
    entry = await _setup_entry(hass, model_bytes)
    new_model = build_teachable_machine_like(num_classes=2, seed=7, size=32)
    archive = make_zip(tmp_path / "new.zip", new_model, ["Car", "Vuoto"])

    result = await hass.config_entries.options.async_init(entry.entry_id)
    with mock_upload(archive):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            _options_input(**{CONF_MODEL_FILE: "6f1c2a52-5c1e-4a43-9a8e-0d6f1b7a1a02"}),
        )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "classes"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_EMPTY_CLASS: "Vuoto", CONF_TARGET_CLASS: "Car"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_EMPTY_CLASS] == "Vuoto"
    assert entry.options[CONF_TARGET_CLASS] == "Car"
    assert entry.options[CONF_MODEL_SHA256] == Classifier(new_model, ["Car", "Vuoto"]).sha256
    assert entry.runtime_data.labels == ["Car", "Vuoto"]
    assert hass.states.get("binary_sensor.parcheggio_car_present") is not None


@pytest.mark.usefixtures("camera_image")
async def test_options_flow_rejects_bad_model(
    hass: HomeAssistant, tmp_path: Path, model_bytes: bytes
) -> None:
    """A quantized replacement model is refused and nothing changes."""
    entry = await _setup_entry(hass, model_bytes)
    archive = make_zip(tmp_path / "q.zip", build_quantized_model(), ["a", "b"])
    result = await hass.config_entries.options.async_init(entry.entry_id)
    with mock_upload(archive):
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            _options_input(**{CONF_MODEL_FILE: "6f1c2a52-5c1e-4a43-9a8e-0d6f1b7a1a03"}),
        )
    assert result["errors"] == {CONF_MODEL_FILE: "unsupported_quantized"}
    assert entry.runtime_data.labels == LABELS


@pytest.mark.usefixtures("camera_image")
async def test_options_flow_camera_and_watched_attribute(
    hass: HomeAssistant, model_bytes: bytes
) -> None:
    """The camera can be changed; watching a missing attribute is refused."""
    entry = await _setup_entry(hass, model_bytes)
    hass.states.async_set("camera.live_view", "idle", {"video_url": "x"})
    hass.states.async_set("camera.snapshot", "idle", {"timestamp": 5})

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        _options_input(
            **{CONF_CAMERA: "camera.live_view", CONF_UPDATE_MODE: UPDATE_MODE_ATTRIBUTE}
        ),
    )
    assert result["errors"] == {CONF_WATCH_ATTRIBUTE: "attribute_missing"}
    assert result["description_placeholders"]["camera"] == "camera.live_view"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        _options_input(**{CONF_CAMERA: "camera.snapshot", CONF_UPDATE_MODE: UPDATE_MODE_ATTRIBUTE}),
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_CAMERA] == "camera.snapshot"
    assert entry.runtime_data.settings.camera_entity_id == "camera.snapshot"
    # snapshot_time comes from the new camera's attribute
    state = hass.states.get("sensor.parcheggio_status")
    assert state.attributes["snapshot_time"] == "1970-01-01T00:00:05+00:00"
