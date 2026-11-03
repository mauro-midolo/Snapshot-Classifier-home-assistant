"""Constants for the Snapshot Classifier integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "snapshot_classifier"

# Config entry data
CONF_CAMERA: Final = "camera_entity_id"
CONF_STORAGE_ID: Final = "storage_id"

# Config flow fields
CONF_MODEL_FILE: Final = "model_file"

# Options
CONF_EMPTY_CLASS: Final = "empty_class"
CONF_TARGET_CLASS: Final = "target_class"
CONF_THRESHOLD: Final = "threshold"
CONF_UPDATE_MODE: Final = "update_mode"
CONF_WATCH_ATTRIBUTE: Final = "watch_attribute"
CONF_SCAN_INTERVAL: Final = "scan_interval"
CONF_CROP: Final = "crop"
CONF_CROP_X: Final = "x"
CONF_CROP_Y: Final = "y"
CONF_CROP_WIDTH: Final = "width"
CONF_CROP_HEIGHT: Final = "height"
CONF_UNCERTAIN_DIR: Final = "uncertain_snapshot_dir"
CONF_MODEL_SHA256: Final = "model_sha256"

UPDATE_MODE_ATTRIBUTE: Final = "attribute"
UPDATE_MODE_INTERVAL: Final = "interval"
UPDATE_MODE_MANUAL: Final = "manual"
UPDATE_MODES: Final = [UPDATE_MODE_ATTRIBUTE, UPDATE_MODE_INTERVAL, UPDATE_MODE_MANUAL]

DEFAULT_THRESHOLD: Final = 70.0
DEFAULT_WATCH_ATTRIBUTE: Final = "timestamp"
DEFAULT_SCAN_INTERVAL: Final = 60
MIN_SCAN_INTERVAL: Final = 5

# The probability of the "empty" class under which the spot is occupied, and
# the probability of the target class over which it is reported as present.
OCCUPANCY_PROBABILITY: Final = 0.5

STATE_UNCERTAIN: Final = "uncertain"
EVENT_UNCERTAIN: Final = "uncertain"

# Files inside the model storage directory
MODEL_FILENAME: Final = "model.tflite"
LABELS_FILENAME: Final = "labels.txt"

# Service
SERVICE_EXPORT_CROPS: Final = "export_crops"
ATTR_CONFIG_ENTRY_ID: Final = "config_entry_id"
ATTR_SOURCE: Final = "source"
ATTR_DESTINATION: Final = "destination"
ATTR_OVERWRITE: Final = "overwrite"
# Service crop fields. Not "x"/"y": YAML 1.1 parsers (e.g. the frontend
# YAML editor) read a bare ``y`` key as the boolean ``true``.
ATTR_LEFT: Final = "left"
ATTR_TOP: Final = "top"
ATTR_WIDTH: Final = "width"
ATTR_HEIGHT: Final = "height"

IMAGE_EXTENSIONS: Final = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
