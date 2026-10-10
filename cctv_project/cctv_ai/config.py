"""Central settings. Override any value by creating a settings.json next to app.py."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Models
    person_model_path: str = "yolov5s.pt"   # pretrained COCO model (class 0 = person)
    weapon_model_path: str = "best.pt"      # your custom weapon model (not in the repo!)
    person_conf: float = 0.35
    weapon_conf: float = 0.30

    # Performance
    detect_every_n_frames: int = 2          # run detection on every Nth frame
    max_process_width: int = 960            # frames wider than this are downscaled

    # Crowd management: people-count thresholds for each level
    crowd_medium: int = 10
    crowd_high: int = 25
    crowd_critical: int = 50

    # Work monitoring
    absent_alert_sec: float = 30.0          # alert if a work zone is empty this long

    # Alerts / output
    alert_cooldown_sec: float = 15.0        # min gap between identical alerts
    output_dir: str = "outputs"
    zones_file: str = "zones.json"
    show_heatmap: bool = False

    @classmethod
    def load(cls, path: str = "settings.json") -> "Settings":
        s = cls()
        p = Path(path)
        if p.exists():
            try:
                for k, v in json.loads(p.read_text()).items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception as e:  # bad JSON should not stop the app
                print(f"Could not read {path}: {e}")
        return s
