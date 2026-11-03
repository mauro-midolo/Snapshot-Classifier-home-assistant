<p align="center">
  <img src="custom_components/snapshot_classifier/brand/icon.png" width="96" alt="Snapshot Classifier icon">
</p>

# Snapshot Classifier for Home Assistant

[![Validate](https://github.com/mauro-midolo/snapshot-classifier-home-assistant/actions/workflows/validate.yml/badge.svg)](https://github.com/mauro-midolo/snapshot-classifier-home-assistant/actions/workflows/validate.yml)
[![Tests](https://github.com/mauro-midolo/snapshot-classifier-home-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/mauro-midolo/snapshot-classifier-home-assistant/actions/workflows/tests.yml)
[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz/docs/faq/custom_repositories)

🇮🇹 [Leggi in italiano](README.it.md)

Classify the snapshot of any Home Assistant camera with an image model you train
yourself, in the browser, with
[Google Teachable Machine](https://teachablemachine.withgoogle.com/). No cloud,
no add-on and no TensorFlow: the model runs locally inside Home Assistant.

Typical use: *"In my parking spot is there **my car**, **another car**, or is it
**empty**?"*. But any question you can answer by looking at one picture works:
"is the garage door open?", "are the bins out?", "is the gate closed?".

## Features

- Setup entirely from the UI: pick the camera, upload the `.zip` exported by
  Teachable Machine, choose which class means "empty".
- Entities for each classifier: **Status** (best class or *Uncertain*),
  **Confidence**, **Occupied**, **&lt;class&gt; present**, **Analyze now** button and
  an **Uncertain result** event.
- Analyzes a new snapshot when a camera attribute changes (e.g. `timestamp` of
  ring-mqtt cameras), at a fixed interval, or only on demand.
- Optional crop, so the model only looks at the part of the picture that matters,
  plus a service to crop your training pictures exactly the same way.
- Optionally saves uncertain snapshots to a folder, so you know which pictures to
  add to the training set.
- Replace the model with a retrained one from the options, without losing entities.
- As many classifiers as you like (several cameras, or several spots on the same
  camera using different crops).
- Works on every Home Assistant installation, including Home Assistant OS on
  Raspberry Pi / Home Assistant Yellow / Green (aarch64): inference is pure
  `numpy`, the only requirement is the pure-Python
  [`tflite`](https://pypi.org/project/tflite/) flatbuffer parser.

## Installation

### HACS (recommended)

1. HACS → ⋮ → **Custom repositories** → add
   `https://github.com/mauro-midolo/snapshot-classifier-home-assistant`, type
   **Integration**.
2. Search for **Snapshot Classifier** in HACS and download it.
3. Restart Home Assistant.

### Manual

Copy `custom_components/snapshot_classifier` into the `custom_components` folder of
your Home Assistant configuration and restart.

## 1. Collect training pictures

The model is only as good as its pictures. They must come **from the same camera,
with the same framing**, in all the situations you want to recognise.

Save a snapshot periodically for a few days (day, night, rain, sun, headlights, …).
For example every 15 minutes into `/media/parking`:

```yaml
alias: Parking - save training snapshot
triggers:
  - trigger: time_pattern
    minutes: "/15"
actions:
  - action: camera.snapshot
    target:
      entity_id: camera.my_camera
    data:
      filename: "/media/parking/{{ now().strftime('%Y%m%d_%H%M%S') }}.jpg"
mode: single
```

> Battery cameras (e.g. Ring via ring-mqtt) only update their snapshot from time
> to time; ask for a fresh one first if your camera supports it.

Download the pictures to your computer (Media browser, Samba share, SSH …) and sort
them into one folder per class, e.g. `my_car/`, `other_car/`, `empty/`.

### Optional: crop

If the camera sees much more than what matters (e.g. the parking spot is only the
bottom-left part of the frame), use a crop: it improves accuracy a lot.

- The crop is a box in fractions of the image: `x`, `y` (top-left corner),
  `width`, `height`, all between 0 and 1. Example: the bottom-left quarter is
  `x: 0, y: 0.5, width: 0.5, height: 0.5`. Measure it on a snapshot with any image
  viewer: `x = left_pixel / image_width`, etc.
- The crop is applied **before** the centre square crop and the resize to 224×224
  that Teachable Machine does. So the box can be any rectangle, but what you want
  to see should be in its centre square.
- **The training pictures must be cropped exactly the same way.** Use the
  `snapshot_classifier.export_crops` service:

```yaml
action: snapshot_classifier.export_crops
data:
  source: /media/parking
  destination: /media/parking_crops
  x: 0
  y: 0.5
  width: 0.5
  height: 0.5
```

Once the classifier exists you can use `config_entry_id` instead of the numbers to
reuse its crop. The service returns how many pictures it exported, skipped
(already present, unless `overwrite: true`) and failed.

## 2. Train the model on Teachable Machine

1. Open <https://teachablemachine.withgoogle.com/> and click **Get Started**.
2. Choose **Image Project** → **Standard image model** (not "Embedded image model").
3. Create **one class per situation** and rename it by clicking on the name
   ("Class 1" → `My car`, "Class 2" → `Other car`, add a class `Empty`). The class
   names become the states of the *Status* sensor, so pick names you like to see
   in Home Assistant.
4. For every class click **Upload** → **Choose images from your files** (or drag
   them) and add the pictures of that class (cropped ones, if you use a crop).
   Don't use the webcam button: the pictures must come from your camera.
5. Tips for a good model:
   - Keep the classes **balanced**: a similar number of pictures in each class
     (at least 30–50 each; more is better). A class with very few pictures will
     rarely be predicted.
   - Cover all conditions: day/night (infrared), rain, shadows, snow, people
     walking by, the car parked slightly differently, doors open…
   - Don't put the same picture in two classes, and leave ambiguous pictures out.
6. Click **Train Model** and wait (keep the browser tab in the foreground). The
   defaults (50 epochs, batch size 16, learning rate 0.001) are fine; under
   **Advanced** you can raise the epochs if the results are poor.
7. Test it in the **Preview** panel: switch Input to **File** and drop some pictures
   that were *not* used for training.
8. Click **Export Model** → **Tensorflow Lite** tab → Model conversion type
   **Floating point** → **Download my model**. Conversion takes up to a minute;
   you get `converted_tflite.zip` with `model_unquant.tflite` and `labels.txt`.
   > ⚠️ "Quantized" and "EdgeTPU" models are **not supported**; the setup will
   > tell you to export in floating point.
9. Optionally **save the project** (☰ → *Save project to Drive* or *Download
   project as file*) so you can add pictures and retrain later.

## 3. Add the integration

**Settings → Devices & services → Add integration → Snapshot Classifier**:

1. **Name** (e.g. `Parking`), **Camera**, **Teachable Machine model** (upload the
   `converted_tflite.zip`).
2. **Classes** (both optional):
   - *Class meaning "empty"* → creates the **Occupied** sensor.
   - *Class to track* (e.g. `My car`) → creates the **My car present** sensor.

If the camera has a `timestamp` attribute the classifier analyzes every new snapshot,
otherwise it polls every 60 seconds. Change it in the options.

## Options

| Option | Default | Description |
| --- | --- | --- |
| Confidence threshold | 70 % | Under this probability *Status* is `Uncertain`. |
| When to analyze | camera attribute if `timestamp` exists, else interval | *When the camera attribute changes*, *At a fixed interval*, *Only manually*. |
| Camera attribute to watch | `timestamp` | Attribute whose change means "new snapshot". Don't use `entity_picture`/`access_token`: it changes every 5 minutes even without a new image. |
| Interval | 60 s | Polling interval. Identical images are not analyzed twice. |
| Class meaning "empty" / Class to track | – | See above. |
| Crop | whole image | `x`, `y`, `width`, `height` in 0–1. |
| Folder for uncertain snapshots | – | e.g. `/media/snapshot_classifier/uncertain`. Must be an allowed folder (`/media` is allowed by default, others go in [`allowlist_external_dirs`](https://www.home-assistant.io/integrations/homeassistant/#allowlist_external_dirs)). |
| Replace the model | – | Upload a retrained `.zip`. If the classes changed you are asked to choose them again. |

## Entities

For a classifier named *Parking*:

| Entity | Description |
| --- | --- |
| `sensor.parking_status` | Best class, or `uncertain` under the threshold. Attributes: `probabilities` (% per class), `best_class`, `threshold`, `snapshot_time`, `analyzed_at`, `inference_seconds`. |
| `sensor.parking_confidence` | Probability of the best class (%). |
| `binary_sensor.parking_occupied` | `on` when the "empty" class has a probability below 50 %. |
| `binary_sensor.parking_my_car_present` | `on` when the tracked class has a probability of 50 % or more. |
| `button.parking_analyze_now` | Analyze the current snapshot (even if unchanged). |
| `event.parking_uncertain_result` | Fires `uncertain` with `best_class`, `confidence`, `probabilities`, `snapshot_time` and `image_path` (if saved). |

Occupied / present use the raw probabilities, so they keep working when the status
is uncertain (e.g. 45 % my car, 40 % other car → occupied, but my car not present).

Example: notify uncertain results to improve the training set.

```yaml
alias: Parking - uncertain snapshot
triggers:
  - trigger: state
    entity_id: event.parking_uncertain_result
    not_from: unavailable
actions:
  - action: notify.mobile_app_my_phone
    data:
      message: >
        Not sure: {{ trigger.to_state.attributes.best_class }}
        ({{ trigger.to_state.attributes.confidence }} %)
```

## Retraining

Add the new pictures (for example from the uncertain folder) to your Teachable
Machine project, train, export again in floating point and upload the zip from
**Settings → Devices & services → Snapshot Classifier → Configure → Replace the
model**.

## How it works

- Image from the camera (`camera.async_get_image`) → crop → centre square →
  bilinear resize to the model input (224×224) → pixels scaled to [-1, 1] → model →
  probabilities. This matches Teachable Machine's own preprocessing.
- TensorFlow Lite / LiteRT and ONNX Runtime have no wheels for the Alpine (musl)
  Python of Home Assistant, so the integration includes a small TFLite interpreter
  written in `numpy`. It supports the operators of Teachable Machine image models
  (`PAD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MEAN`, `FULLY_CONNECTED`,
  `SOFTMAX`) and a few more (`MUL`, `RESHAPE`, `CONCATENATION`, pooling,
  activations). The tests compare it with the official LiteRT runtime.
- The model is parsed once at startup; inference (≈30 ms on a desktop CPU, well
  under a second on a Raspberry Pi 4/CM4) runs in the executor, never in the event
  loop.
- The uploaded model is stored in `config/.storage/snapshot_classifier/<id>/`
  (included in backups, deleted when the integration is removed).

## Troubleshooting

- **"Quantized model"**: export again choosing *Floating point*.
- **Always uncertain / wrong class**: more and more balanced pictures; make sure
  the training pictures have the same crop as the classifier.
- **Never updates**: check the *When to analyze* option. With a camera attribute,
  look in Developer tools → States which attribute changes with each new snapshot.
- Debug logs:

  ```yaml
  logger:
    logs:
      custom_components.snapshot_classifier: debug
  ```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_test.txt
pytest
```

`SNAPSHOT_CLASSIFIER_TEST_MODEL=/path/to/model_unquant.tflite pytest` also compares a
real Teachable Machine export with LiteRT.

## License

[MIT](LICENSE). Teachable Machine is a Google product; this project is not
affiliated with or endorsed by Google.
