# CCTV AI: Crowd Management, Crime Prevention & Work Monitoring

A desktop application that turns existing CCTV feeds into live analytics using YOLO object detection. It works with image files, video files, webcams and IP / RTSP cameras.

## Features

| Area | What it does |
|---|---|
| **Crowd management** | Counts people, classifies the crowd as LOW / MEDIUM / HIGH / CRITICAL, shows an optional density heatmap, and raises overcrowding alerts |
| **Crime prevention** | Detects weapons using your custom model, alerts when someone enters a restricted zone, and saves a snapshot of every serious alert |
| **Work monitoring** | Tracks how long work zones (desks, counters) are occupied or empty, and alerts when a zone stays empty too long |
| **Live sources** | Image, video file, webcam, IP / RTSP camera (reconnects automatically if the stream drops) |
| **Crowd Graph** | One click plots people count over time for the latest session, with crowd thresholds and weapon detections marked |
| **Logging** | Alerts and crowd data are saved to CSV files, with snapshots of serious alerts |

All analysis is anonymous. The app uses only counts and positions, with no face recognition or personal identification.

## Requirements

- Python 3.9 or newer (tested layout for 3.11)
- Windows, Linux or macOS with a display (the app uses a Tkinter window)
- Internet access on the first run, so the YOLOv5 code and any fallback model can be downloaded once
- A GPU is optional. It runs on CPU, but more slowly.

## Installation

1. Extract the project and open a terminal **inside the folder that contains `app.py`**. If you extracted a zip, this is usually a folder one level deeper than where you started.

   ```powershell
   dir
   ```

   You should see `app.py`, `requirements.txt`, `yolov5s.pt` and the `cctv_ai` folder.

2. (Recommended) create a virtual environment:

   ```powershell
   python -m venv venv
   venv\Scripts\activate          # Linux / macOS: source venv/bin/activate
   ```

3. Install the dependencies. This can take several minutes because PyTorch is a large download:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   python app.py
   ```

## Quick start

1. Click **Load Models**. The log shows which person model loaded and whether the weapon model was found.
2. Pick a source:
   - **Image Detection** for a single picture
   - **Video File** for a recorded video
   - **Webcam** for a connected camera
   - **IP / RTSP Camera** and enter a URL such as `rtsp://user:password@192.168.1.10:554/stream1`
3. Watch the live view and the **Live Analytics** panel on the right. Click **Stop** to end a session. A summary is printed to the log.
4. Click **Crowd Graph** to plot the people count over time for the most recent session.

## Defining zones

Zones let the app watch specific areas of the camera view.

1. Click **Add Zone** and choose an image or video from the camera you want to monitor.
2. Draw a rectangle on the frame and press **Enter** (press **C** to cancel).
3. Name the zone, for example `Desk-1` or `Server Room`.
4. Choose the type:
   - **Work zone**: tracks time present and away, and alerts if the zone is empty longer than `absent_alert_sec`.
   - **Restricted zone**: raises an alert whenever someone is inside.

Zones are saved to `zones.json` and apply to every source. Positions are stored as fractions of the frame, so they work at any resolution. Click **Clear Zones** to delete them all.

## Models

| File | Purpose | Included? |
|---|---|---|
| `yolov5s.pt` | Person detection (pretrained COCO model) | Yes |
| `best.pt` | Weapon detection (your custom-trained model) | **No, you must add it** |

**Person model.** The app first tries to load `yolov5s.pt` through `torch.hub`. If that fails, it falls back to a standard Ultralytics model (`yolov5su.pt` or `yolov8n.pt`). The log shows which one was used.

**Weapon model.** Place your trained `best.pt` next to `app.py`. Class names are read from the model itself, so no code changes are needed. Without this file, everything else works but weapon detection is disabled.

## Settings

Create a file named `settings.json` next to `app.py` to override any default. Example:

```json
{
  "crowd_medium": 5,
  "crowd_high": 15,
  "crowd_critical": 30,
  "absent_alert_sec": 60,
  "detect_every_n_frames": 3
}
```

| Setting | Default | Meaning |
|---|---|---|
| `person_model_path` | `yolov5s.pt` | Person detection weights |
| `weapon_model_path` | `best.pt` | Weapon detection weights |
| `person_conf` | 0.35 | Minimum confidence for people |
| `weapon_conf` | 0.30 | Minimum confidence for weapons |
| `detect_every_n_frames` | 2 | Run detection on every Nth frame. Raise it if playback is slow |
| `max_process_width` | 960 | Frames wider than this are downscaled for speed |
| `crowd_medium` / `crowd_high` / `crowd_critical` | 10 / 25 / 50 | People counts where each crowd level begins |
| `absent_alert_sec` | 30 | Seconds a work zone may be empty before an alert |
| `alert_cooldown_sec` | 15 | Minimum gap between repeats of the same alert |
| `output_dir` | `outputs` | Where logs, snapshots and graphs are saved |
| `zones_file` | `zones.json` | Where zones are stored |

## Output files

Everything is written to the `outputs` folder:

| File | Contents |
|---|---|
| `alerts.csv` | Every alert: time, severity, message, snapshot path |
| `crowd_log.csv` | One row per second: people count, crowd level, weapons detected |
| `snapshots/` | Annotated frame for every HIGH or CRITICAL alert |
| `crowd_graph.png` | Latest graph from the **Crowd Graph** button |
| `annotated_<name>.jpg` | Result of image detection |

## Project structure

```
app.py                 GUI (Tkinter)
requirements.txt       Python dependencies
yolov5s.pt             Person detection weights
best.pt                Weapon detection weights (you provide)
settings.json          Optional overrides
zones.json             Created when you add zones
cctv_ai/
    config.py          Settings and defaults
    detectors.py       Person and weapon model wrappers
    analytics.py       Crowd levels, heatmap, zone monitoring
    alerts.py          Alert cooldown, CSV log, snapshots
    pipeline.py        Frame analyzer and threaded stream processor
outputs/               Created automatically
```

## Troubleshooting

**`Could not open requirements file`**
You are in the wrong folder. Use `dir` to check that `requirements.txt` is in the current folder, and `cd` into the inner project folder if needed.

**`Import "ultralytics" could not be resolved` in VS Code**
VS Code is using a different Python than the one you installed packages into. Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the Python you used with pip. Check the install with `python -c "import ultralytics; print(ultralytics.__version__)"`.

**Person model fails to load**
`torch.hub` needs internet the first time. If it still fails, the app falls back to a standard YOLO model. The log lists the errors.

**`Weapon model not loaded`**
`best.pt` is missing from the project folder. Add your trained file and click **Load Models** again.

**Video is slow or choppy**
Raise `detect_every_n_frames` to 3 or 4, or lower `max_process_width`. A GPU helps significantly.

**RTSP camera will not open**
Test the URL in VLC first. Check the username, password, port and stream path for your camera model. The app retries a lost stream up to 5 times.

**Crowd Graph says there is no data**
Run a video or camera first. The graph is built from `outputs/crowd_log.csv`.

## Limitations

- Weapon detection is only as accurate as your `best.pt` and its training data. Expect some false alarms and missed detections. Treat alerts as "a person should review this", not as proof.
- Zones use the centre of each person's bounding box and do not track individual identities. "Present" means someone is in the zone, not a specific person.
- Counting is detection-based and becomes less accurate in very dense crowds where people overlap heavily.
- Heavy video processing is demanding on CPU-only machines.

## Privacy and legal notice

CCTV analytics, and especially monitoring of employees, may be regulated where you live or work. Before deploying this on real cameras, check local laws and workplace policies on surveillance, signage, consent, data retention and access to recordings. Keep alert snapshots and logs secure, and delete them when they are no longer needed.
