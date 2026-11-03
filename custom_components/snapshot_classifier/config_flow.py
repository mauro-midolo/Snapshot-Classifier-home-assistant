"""Config and options flow for Snapshot Classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import voluptuous as vol

from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_NAME, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    FileSelector,
    FileSelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from . import model_directory
from .classifier import (
    Classifier,
    CropBox,
    ModelArchiveError,
    parse_labels,
    read_model_archive,
    save_model,
)
from .const import (
    CONF_CAMERA,
    CONF_CROP,
    CONF_CROP_HEIGHT,
    CONF_CROP_WIDTH,
    CONF_CROP_X,
    CONF_CROP_Y,
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
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_THRESHOLD,
    DEFAULT_WATCH_ATTRIBUTE,
    DOMAIN,
    LABELS_FILENAME,
    MIN_SCAN_INTERVAL,
    UPDATE_MODE_ATTRIBUTE,
    UPDATE_MODE_INTERVAL,
    UPDATE_MODES,
)

MODEL_SELECTOR = FileSelector(FileSelectorConfig(accept=".zip,application/zip"))
FRACTION_SELECTOR = NumberSelector(
    NumberSelectorConfig(min=0, max=1, step=0.001, mode=NumberSelectorMode.BOX)
)


async def _async_read_upload(
    hass: HomeAssistant, file_id: str
) -> tuple[Classifier | None, dict[str, str], dict[str, str]]:
    """Validate an uploaded archive. Returns (classifier, errors, placeholders)."""

    def _read() -> Classifier:
        with process_uploaded_file(hass, file_id) as path:
            return read_model_archive(path)

    try:
        return await hass.async_add_executor_job(_read), {}, {}
    except ModelArchiveError as err:
        return None, {CONF_MODEL_FILE: err.reason}, dict(err.placeholders)
    except ValueError:  # unknown/expired upload id
        return None, {CONF_MODEL_FILE: "invalid_archive"}, {}


def _classes_schema(labels: list[str]) -> vol.Schema:
    selector = SelectSelector(
        SelectSelectorConfig(options=labels, mode=SelectSelectorMode.DROPDOWN)
    )
    return vol.Schema(
        {
            vol.Optional(CONF_EMPTY_CLASS): selector,
            vol.Optional(CONF_TARGET_CLASS): selector,
        }
    )


def _guess_empty_class(labels: list[str]) -> str | None:
    """Suggest the class meaning 'empty' from common names."""
    names = {"vuoto", "vuota", "empty", "none", "nothing", "libero", "free", "leer", "vide"}
    return next((label for label in labels if label.strip().lower() in names), None)


def _classes_from_input(user_input: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    return {
        CONF_EMPTY_CLASS: user_input.get(CONF_EMPTY_CLASS)
        if user_input.get(CONF_EMPTY_CLASS) in labels
        else None,
        CONF_TARGET_CLASS: user_input.get(CONF_TARGET_CLASS)
        if user_input.get(CONF_TARGET_CLASS) in labels
        else None,
    }


class SnapshotClassifierConfigFlow(ConfigFlow, domain=DOMAIN):
    """Create a classifier for one camera."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the flow."""
        self._classifier: Classifier | None = None
        self._name = ""
        self._camera = ""

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SnapshotClassifierOptionsFlow:
        """Return the options flow."""
        return SnapshotClassifierOptionsFlow()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Choose the camera and upload the Teachable Machine export."""
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {}
        if user_input is not None:
            classifier, errors, placeholders = await _async_read_upload(
                self.hass, user_input[CONF_MODEL_FILE]
            )
            if classifier is not None:
                self._classifier = classifier
                self._name = user_input[CONF_NAME].strip()
                self._camera = user_input[CONF_CAMERA]
                return await self.async_step_classes()

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): TextSelector(),
                vol.Required(CONF_CAMERA): EntitySelector(EntitySelectorConfig(domain="camera")),
                vol.Required(CONF_MODEL_FILE): MODEL_SELECTOR,
            }
        )
        if user_input is not None:
            schema = self.add_suggested_values_to_schema(
                schema, {k: v for k, v in user_input.items() if k != CONF_MODEL_FILE}
            )
        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
            description_placeholders=placeholders,
        )

    async def async_step_classes(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Choose which class means 'empty' and which one to track."""
        assert self._classifier is not None
        labels = self._classifier.labels
        if user_input is not None:
            return await self._async_create(_classes_from_input(user_input, labels))

        schema = self.add_suggested_values_to_schema(
            _classes_schema(labels), {CONF_EMPTY_CLASS: _guess_empty_class(labels)}
        )
        return self.async_show_form(
            step_id="classes",
            data_schema=schema,
            description_placeholders={"classes": ", ".join(labels)},
        )

    async def _async_create(self, classes: dict[str, Any]) -> ConfigFlowResult:
        assert self._classifier is not None
        classifier = self._classifier
        storage_id = uuid.uuid4().hex
        await self.hass.async_add_executor_job(
            save_model, model_directory(self.hass, storage_id), classifier
        )

        camera_state = self.hass.states.get(self._camera)
        has_timestamp = (
            camera_state is not None and DEFAULT_WATCH_ATTRIBUTE in camera_state.attributes
        )
        options = {
            CONF_THRESHOLD: DEFAULT_THRESHOLD,
            CONF_UPDATE_MODE: UPDATE_MODE_ATTRIBUTE if has_timestamp else UPDATE_MODE_INTERVAL,
            CONF_WATCH_ATTRIBUTE: DEFAULT_WATCH_ATTRIBUTE,
            CONF_SCAN_INTERVAL: DEFAULT_SCAN_INTERVAL,
            CONF_CROP: CropBox().as_dict(),
            CONF_MODEL_SHA256: classifier.sha256,
            **{k: v for k, v in classes.items() if v},
        }
        return self.async_create_entry(
            title=self._name,
            data={CONF_CAMERA: self._camera, CONF_STORAGE_ID: storage_id},
            options=options,
        )


class SnapshotClassifierOptionsFlow(OptionsFlow):
    """Change thresholds, triggers, crop and replace the model."""

    def __init__(self) -> None:
        """Initialize the flow."""
        self._options: dict[str, Any] = {}
        self._new_classifier: Classifier | None = None

    @property
    def _camera(self) -> str:
        return self.config_entry.options.get(CONF_CAMERA) or self.config_entry.data[CONF_CAMERA]

    @property
    def _directory(self) -> Path:
        return model_directory(self.hass, self.config_entry.data[CONF_STORAGE_ID])

    async def _async_current_labels(self) -> list[str]:
        path = self._directory / LABELS_FILENAME
        try:
            text = await self.hass.async_add_executor_job(path.read_text, "utf-8")
        except OSError:
            return []
        return parse_labels(text)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Main options form."""
        labels = await self._async_current_labels()
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {}

        if user_input is not None:
            crop = CropBox.from_mapping(user_input.get(CONF_CROP))
            uncertain_dir = (user_input.get(CONF_UNCERTAIN_DIR) or "").strip()
            if not crop.is_valid:
                errors[CONF_CROP] = "invalid_crop"
            if uncertain_dir and not (
                Path(uncertain_dir).is_absolute()
                and self.hass.config.is_allowed_path(uncertain_dir)
            ):
                errors[CONF_UNCERTAIN_DIR] = "path_not_allowed"
            camera = user_input.get(CONF_CAMERA) or self._camera
            attribute = (user_input.get(CONF_WATCH_ATTRIBUTE) or DEFAULT_WATCH_ATTRIBUTE).strip()
            if user_input[CONF_UPDATE_MODE] == UPDATE_MODE_ATTRIBUTE and (
                (state := self.hass.states.get(camera)) is not None
                and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
                and attribute not in state.attributes
            ):
                # Watching an attribute the camera does not have means the
                # classifier would never run on its own.
                errors[CONF_WATCH_ATTRIBUTE] = "attribute_missing"
                placeholders.update({"camera": camera, "attribute": attribute})
            if not errors and (file_id := user_input.get(CONF_MODEL_FILE)):
                self._new_classifier, errors, placeholders = await _async_read_upload(
                    self.hass, file_id
                )
            if not errors:
                self._options = {
                    **self.config_entry.options,
                    CONF_CAMERA: camera,
                    CONF_THRESHOLD: float(user_input[CONF_THRESHOLD]),
                    CONF_UPDATE_MODE: user_input[CONF_UPDATE_MODE],
                    CONF_WATCH_ATTRIBUTE: attribute,
                    CONF_SCAN_INTERVAL: int(user_input[CONF_SCAN_INTERVAL]),
                    CONF_CROP: crop.as_dict(),
                    CONF_UNCERTAIN_DIR: uncertain_dir,
                    CONF_EMPTY_CLASS: user_input.get(CONF_EMPTY_CLASS),
                    CONF_TARGET_CLASS: user_input.get(CONF_TARGET_CLASS),
                }
                if self._new_classifier is not None:
                    new_labels = self._new_classifier.labels
                    chosen = [
                        self._options[key]
                        for key in (CONF_EMPTY_CLASS, CONF_TARGET_CLASS)
                        if self._options.get(key)
                    ]
                    if new_labels != labels or any(c not in new_labels for c in chosen):
                        return await self.async_step_classes()
                return await self._async_finish()

        options = self.config_entry.options
        crop_defaults = CropBox.from_mapping(options.get(CONF_CROP)).as_dict()
        label_selector = SelectSelector(
            SelectSelectorConfig(options=labels, mode=SelectSelectorMode.DROPDOWN)
        )
        schema = vol.Schema(
            {
                vol.Required(CONF_CAMERA): EntitySelector(EntitySelectorConfig(domain="camera")),
                vol.Required(CONF_THRESHOLD): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=100,
                        step=1,
                        unit_of_measurement="%",
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(CONF_UPDATE_MODE): SelectSelector(
                    SelectSelectorConfig(
                        options=UPDATE_MODES,
                        translation_key=CONF_UPDATE_MODE,
                        mode=SelectSelectorMode.LIST,
                    )
                ),
                vol.Optional(CONF_WATCH_ATTRIBUTE): TextSelector(),
                vol.Required(CONF_SCAN_INTERVAL): NumberSelector(
                    NumberSelectorConfig(
                        min=MIN_SCAN_INTERVAL,
                        max=86400,
                        step=1,
                        unit_of_measurement="s",
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_EMPTY_CLASS): label_selector,
                vol.Optional(CONF_TARGET_CLASS): label_selector,
                vol.Required(CONF_CROP): section(
                    vol.Schema(
                        {
                            vol.Required(CONF_CROP_X): FRACTION_SELECTOR,
                            vol.Required(CONF_CROP_Y): FRACTION_SELECTOR,
                            vol.Required(CONF_CROP_WIDTH): FRACTION_SELECTOR,
                            vol.Required(CONF_CROP_HEIGHT): FRACTION_SELECTOR,
                        }
                    ),
                    {"collapsed": crop_defaults == CropBox().as_dict()},
                ),
                vol.Optional(CONF_UNCERTAIN_DIR): TextSelector(),
                vol.Optional(CONF_MODEL_FILE): MODEL_SELECTOR,
            }
        )
        suggested = {
            CONF_CAMERA: self._camera,
            CONF_THRESHOLD: options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            CONF_UPDATE_MODE: options.get(CONF_UPDATE_MODE, UPDATE_MODE_ATTRIBUTE),
            CONF_WATCH_ATTRIBUTE: options.get(CONF_WATCH_ATTRIBUTE, DEFAULT_WATCH_ATTRIBUTE),
            CONF_SCAN_INTERVAL: options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
            CONF_EMPTY_CLASS: options.get(CONF_EMPTY_CLASS),
            CONF_TARGET_CLASS: options.get(CONF_TARGET_CLASS),
            CONF_CROP: crop_defaults,
            CONF_UNCERTAIN_DIR: options.get(CONF_UNCERTAIN_DIR, ""),
        }
        if user_input is not None:
            suggested.update({k: v for k, v in user_input.items() if k != CONF_MODEL_FILE})
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
            description_placeholders={
                "classes": ", ".join(labels),
                "details": "",
                "outputs": "",
                "labels": "",
                "camera": "",
                "attribute": "",
                **placeholders,
            },
        )

    async def async_step_classes(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Choose the classes again after uploading a model with new labels."""
        assert self._new_classifier is not None
        labels = self._new_classifier.labels
        if user_input is not None:
            self._options.update(_classes_from_input(user_input, labels))
            return await self._async_finish()

        suggested = {
            key: value
            for key in (CONF_EMPTY_CLASS, CONF_TARGET_CLASS)
            if (value := self._options.get(key)) in labels
        }
        suggested.setdefault(CONF_EMPTY_CLASS, _guess_empty_class(labels))
        return self.async_show_form(
            step_id="classes",
            data_schema=self.add_suggested_values_to_schema(_classes_schema(labels), suggested),
            description_placeholders={"classes": ", ".join(labels)},
        )

    async def _async_finish(self) -> ConfigFlowResult:
        if self._new_classifier is not None:
            await self.hass.async_add_executor_job(
                save_model, self._directory, self._new_classifier
            )
            self._options[CONF_MODEL_SHA256] = self._new_classifier.sha256
        options = {k: v for k, v in self._options.items() if v not in (None, "")}
        return self.async_create_entry(data=options)
