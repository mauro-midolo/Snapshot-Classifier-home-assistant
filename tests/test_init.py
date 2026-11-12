"""Tests for setup, entities, triggers and services."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.snapshot_classifier import model_directory
from custom_components.snapshot_classifier.classifier import Classifier, save_model
from custom_components.snapshot_classifier.const import (
    CONF_CAMERA,
    CONF_CROP,
    CONF_EMPTY_CLASS,
    CONF_SCAN_INTERVAL,
    CONF_STORAGE_ID,
    CONF_TARGET_CLASS,
    CONF_THRESHOLD,
    CONF_UNCERTAIN_DIR,
    CONF_UPDATE_MODE,
    DOMAIN,
    UPDATE_MODE_ATTRIBUTE,
    UPDATE_MODE_INTERVAL,
    UPDATE_MODE_MANUAL,
)
from homeassistant.components.button import SERVICE_PRESS
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.util import dt as dt_util

from .conftest import LABELS, jpeg

CAMERA = "camera.ingresso"
STATUS = "sensor.parcheggio_status"
CONFIDENCE = "sensor.parcheggio_confidence"
OCCUPIED = "binary_sensor.parcheggio_occupied"
PRESENT = "binary_sensor.parcheggio_mine_present"
BUTTON = "button.parcheggio_analyze_now"
EVENT = "event.parcheggio_uncertain_result"

MINE = {"Mine": 0.9, "Other": 0.06, "Empty": 0.04}
EMPTY = {"Mine": 0.01, "Other": 0.04, "Empty": 0.95}
UNSURE = {"Mine": 0.4, "Other": 0.35, "Empty": 0.25}


async def setup_entry(hass: HomeAssistant, model_bytes: bytes, **options: Any) -> MockConfigEntry:
    """Store the model and set up an entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Parcheggio",
        data={CONF_CAMERA: CAMERA, CONF_STORAGE_ID: "store"},
        options={
            CONF_THRESHOLD: 70,
            CONF_UPDATE_MODE: UPDATE_MODE_ATTRIBUTE,
            CONF_EMPTY_CLASS: "Empty",
            CONF_TARGET_CLASS: "Mine",
            **options,
        },
    )
    entry.add_to_hass(hass)
    await hass.async_add_executor_job(
        save_model, model_directory(hass, "store"), Classifier(model_bytes, LABELS)
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def mock_classify(*results: dict[str, float]):
    """Patch the classifier to return ``results`` in order (last one repeats)."""
    queue = list(results)

    def _classify(self, image, crop):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return patch.object(Classifier, "classify", autospec=True, side_effect=_classify)


async def test_real_inference(hass: HomeAssistant, model_bytes: bytes, camera_image) -> None:
    """End to end with the numpy interpreter: entities get a valid state."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1700000000})
    entry = await setup_entry(hass, model_bytes)
    assert entry.state is ConfigEntryState.LOADED

    state = hass.states.get(STATUS)
    assert state.state in [*LABELS, "uncertain"]
    assert state.attributes["options"] == [*LABELS, "uncertain"]
    probabilities = state.attributes["probabilities"]
    assert set(probabilities) == set(LABELS)
    assert sum(probabilities.values()) == pytest.approx(100, abs=0.05)
    assert state.attributes["snapshot_time"] == "2023-11-14T22:13:20+00:00"
    assert float(hass.states.get(CONFIDENCE).state) == max(probabilities.values())


async def test_entities_and_attribute_trigger(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """A new value of the watched attribute triggers an analysis."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1, "entity_picture": "/a?token=1"})
    with mock_classify(MINE, EMPTY) as classify:
        await setup_entry(hass, model_bytes)
        assert classify.call_count == 1
        assert hass.states.get(STATUS).state == "Mine"
        assert hass.states.get(CONFIDENCE).state == "90.0"
        assert hass.states.get(OCCUPIED).state == STATE_ON
        assert hass.states.get(PRESENT).state == STATE_ON

        # entity_picture/access_token rotation alone must not trigger
        hass.states.async_set(CAMERA, "idle", {"timestamp": 1, "entity_picture": "/a?token=2"})
        await hass.async_block_till_done()
        assert classify.call_count == 1

        camera_image.append(jpeg((10, 200, 10)))
        hass.states.async_set(CAMERA, "idle", {"timestamp": 2, "entity_picture": "/a?token=2"})
        await hass.async_block_till_done()
        assert classify.call_count == 2

    assert hass.states.get(STATUS).state == "Empty"
    assert hass.states.get(OCCUPIED).state == STATE_OFF
    assert hass.states.get(PRESENT).state == STATE_OFF


async def test_unchanged_image_is_not_analyzed_again(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """Same bytes: no new inference; the button forces one."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(MINE) as classify:
        await setup_entry(hass, model_bytes)
        hass.states.async_set(CAMERA, "idle", {"timestamp": 2})
        await hass.async_block_till_done()
        assert classify.call_count == 1

        await hass.services.async_call(
            "button", SERVICE_PRESS, {ATTR_ENTITY_ID: BUTTON}, blocking=True
        )
        assert classify.call_count == 2


async def test_uncertain_event_and_saved_snapshot(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes], tmp_path: Path
) -> None:
    """Under the threshold the state is 'uncertain', an event fires, the image is saved."""
    hass.config.allowlist_external_dirs.add(str(tmp_path))
    folder = tmp_path / "uncertain"
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(UNSURE):
        await setup_entry(hass, model_bytes, **{CONF_UNCERTAIN_DIR: str(folder)})

    assert hass.states.get(STATUS).state == "uncertain"
    event = hass.states.get(EVENT)
    assert event.state != STATE_UNKNOWN
    assert event.attributes["event_type"] == "uncertain"
    assert event.attributes["best_class"] == "Mine"
    assert event.attributes["confidence"] == 40.0
    saved = Path(event.attributes["image_path"])
    assert saved.parent == folder
    assert saved.name.startswith("parcheggio_")
    assert saved.stem.endswith("_mine_40")
    assert saved.read_bytes() == camera_image[-1]
    # Occupancy still works from the raw probabilities
    assert hass.states.get(OCCUPIED).state == STATE_ON
    assert hass.states.get(PRESENT).state == STATE_OFF


async def test_interval_mode(
    hass: HomeAssistant,
    model_bytes: bytes,
    camera_image: list[bytes],
) -> None:
    """Interval mode polls the camera."""
    hass.states.async_set(CAMERA, "idle")
    with mock_classify(MINE, EMPTY) as classify:
        await setup_entry(
            hass,
            model_bytes,
            **{CONF_UPDATE_MODE: UPDATE_MODE_INTERVAL, CONF_SCAN_INTERVAL: 30},
        )
        assert classify.call_count == 1
        camera_image.append(jpeg((1, 2, 3)))
        async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=31))
        await hass.async_block_till_done(wait_background_tasks=True)
        assert classify.call_count == 2
    assert hass.states.get(STATUS).state == "Empty"


async def test_manual_mode(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """Manual mode only analyzes on request."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(EMPTY) as classify:
        await setup_entry(hass, model_bytes, **{CONF_UPDATE_MODE: UPDATE_MODE_MANUAL})
        hass.states.async_set(CAMERA, "idle", {"timestamp": 2})
        await hass.async_block_till_done()
        assert classify.call_count == 0
        assert hass.states.get(STATUS).state == STATE_UNKNOWN

        await hass.services.async_call(
            "button", SERVICE_PRESS, {ATTR_ENTITY_ID: BUTTON}, blocking=True
        )
        assert classify.call_count == 1
    assert hass.states.get(STATUS).state == "Empty"


async def test_optional_binary_sensors(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """Without class choices only the generic entities exist."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(MINE):
        await setup_entry(hass, model_bytes, **{CONF_EMPTY_CLASS: None, CONF_TARGET_CLASS: None})
    assert hass.states.get(OCCUPIED) is None
    assert hass.states.get(PRESENT) is None
    assert hass.states.get(STATUS).state == "Mine"


async def test_crop_is_passed_to_classifier(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """The configured crop reaches the classifier."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    crop = {"x": 0.25, "y": 0.5, "width": 0.5, "height": 0.5}
    with mock_classify(MINE) as classify:
        await setup_entry(hass, model_bytes, **{CONF_CROP: crop})
    assert classify.call_args.args[2].as_dict() == crop


async def test_camera_error_marks_unavailable(hass: HomeAssistant, model_bytes: bytes) -> None:
    """A failing camera makes the sensors unavailable but keeps the entry loaded."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with patch(
        "custom_components.snapshot_classifier.coordinator.camera.async_get_image",
        side_effect=HomeAssistantError("offline"),
    ):
        entry = await setup_entry(hass, model_bytes)
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get(STATUS).state == "unavailable"
    assert hass.states.get(BUTTON).state != "unavailable"


async def test_missing_model_files(hass: HomeAssistant) -> None:
    """Without the stored model the entry fails to load with a clear error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="X",
        data={CONF_CAMERA: CAMERA, CONF_STORAGE_ID: "missing"},
        options={},
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    assert entry.state is ConfigEntryState.SETUP_ERROR


async def test_remove_entry_deletes_model(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes]
) -> None:
    """Removing the entry deletes the stored model."""
    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(MINE):
        entry = await setup_entry(hass, model_bytes)
    directory = model_directory(hass, "store")
    assert directory.is_dir()
    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()
    assert not directory.exists()


async def test_export_crops_service(
    hass: HomeAssistant, model_bytes: bytes, camera_image: list[bytes], tmp_path: Path
) -> None:
    """The service crops a folder of pictures with the entry's crop."""
    hass.config.allowlist_external_dirs.add(str(tmp_path))
    source = tmp_path / "src"
    source.mkdir()
    Image.new("RGB", (200, 100)).save(source / "a.jpg")
    Image.new("RGB", (200, 100)).save(source / "b.png")
    (source / "notes.txt").write_text("ignored")
    (source / "broken.jpg").write_bytes(b"nope")

    hass.states.async_set(CAMERA, "idle", {"timestamp": 1})
    with mock_classify(MINE):
        entry = await setup_entry(
            hass, model_bytes, **{CONF_CROP: {"x": 0.5, "y": 0, "width": 0.5, "height": 0.5}}
        )

    destination = tmp_path / "dst"
    response = await hass.services.async_call(
        DOMAIN,
        "export_crops",
        {"source": str(source), "destination": str(destination), "config_entry_id": entry.entry_id},
        blocking=True,
        return_response=True,
    )
    assert response["exported"] == 2
    assert response["failed"] == ["broken.jpg"]
    with Image.open(destination / "a.jpg") as image:
        assert image.size == (100, 50)
    with Image.open(destination / "b.png") as image:
        assert image.format == "PNG"

    # Explicit fields override; existing files are skipped unless overwrite
    response = await hass.services.async_call(
        DOMAIN,
        "export_crops",
        {"source": str(source), "destination": str(destination), "width": 0.25, "left": 0},
        blocking=True,
        return_response=True,
    )
    assert response["skipped"] == 2

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "export_crops",
            {"source": "/etc", "destination": str(destination)},
            blocking=True,
            return_response=True,
        )
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "export_crops",
            {"source": str(source), "destination": str(destination), "left": 0.9, "width": 0.5},
            blocking=True,
            return_response=True,
        )
