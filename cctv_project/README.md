# CCTV AI: Crowd Management, Crime Prevention & Work Monitoring

Turns existing CCTV feeds (files, webcam, RTSP/IP cameras) into live analytics using YOLO.

## Features
| Area | What it does |
|---|---|
| Crowd management | People count, LOW / MEDIUM / HIGH / CRITICAL level, density heatmap, overcrowding alerts |
| Crime prevention | Weapon detection (your custom model), restricted-zone intrusion alerts, snapshots of every serious alert |
| Work monitoring | Work zones (desks, counters) track present / away time and alert when empty too long |
| Sources | Image, video file, webcam, IP / RTSP camera (auto-reconnect) |
| Output | `outputs/alerts.csv`, `outputs/crowd_log.csv`, `outputs/snapshots/`, annotated images |

Everything is anonymous: only counts and positions are used, no faces or identities.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
1. Click **Load Models** (first run downloads/caches files; internet needed once for the YOLOv5 torch.hub path).
2. Choose **Image**, **Video File**, **Webcam** or **IP / RTSP Camera** (e.g. `rtsp://user:pass@192.168.1.10:554/stream1`).
3. **Add Zone** lets you draw a rectangle on a frame from your camera footage, name it, and mark it *work* or *restricted*. Zones are saved to `zones.json` and apply to every source.

## Models
- **Person model:** `yolov5s.pt` (included). If it can't be loaded, the app falls back to `yolov5su.pt` / `yolov8n.pt`. The log shows which backend loaded.
- **Weapon model:** place your trained `best.pt` next to `app.py`. It is NOT in the GitHub repo. Class labels are read from the model itself (the old hard-coded class list was removed, because wrong index order gives wrong labels). Without it, weapon detection is simply disabled.

## Settings
Create `settings.json` to override defaults from `cctv_ai/config.py`, e.g.:
```json
{"crowd_medium": 5, "crowd_high": 15, "crowd_critical": 30, "absent_alert_sec": 60, "detect_every_n_frames": 3}
```
Raise `detect_every_n_frames` if playback is slow on a CPU.

## Project layout
```
app.py              GUI (Tkinter)
cctv_ai/config.py     settings
cctv_ai/detectors.py  person + weapon model wrappers
cctv_ai/analytics.py  crowd levels, heatmap, zones
cctv_ai/alerts.py     cooldown, CSV log, snapshots
cctv_ai/pipeline.py   frame analyzer + threaded stream processor
```

## Changes from the original app.py
- Detection runs in a background thread; the GUI no longer freezes during video.
- Fixed colour-order bug (RGB was passed to Ultralytics, which expects BGR) and the needless resize-to-640 / rescale step.
- Fixed `.numpy()` crash on GPU tensors; labels now come from the model.
- Added live sources, crowd levels, heatmap, zones, alerts, logging and reconnect.

## Limitations (please read)
- Weapon detection accuracy depends entirely on your `best.pt` and training data; expect false positives and misses. Treat alerts as "needs human review", never as proof.
- Zones use the person's box centre, with no tracking IDs, so "present/away" means "someone is in the zone", not a specific person.
- Counting is detection-based and degrades in very dense crowds (heavy occlusion); for those, a density-map model would be better.
- Check local law and workplace policy on CCTV analytics, signage, retention and employee monitoring before deploying.
