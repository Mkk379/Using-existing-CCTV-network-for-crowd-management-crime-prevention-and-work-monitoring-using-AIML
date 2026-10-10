"""Alert manager: cooldown, CSV log and snapshot of the annotated frame."""
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2


class AlertManager:
    def __init__(self, out_dir, cooldown, on_alert=None):
        self.dir = Path(out_dir)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "alerts.csv"
        self.cooldown = cooldown
        self.on_alert = on_alert
        self.last = {}
        self.count = 0
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "video_time_s", "severity", "message", "snapshot"])

    def raise_alert(self, key, severity, message, t, frame=None):
        if t - self.last.get(key, -1e9) < self.cooldown:
            return False
        self.last[key] = t
        self.count += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = ""
        if frame is not None and severity in ("HIGH", "CRITICAL"):
            snap = str(self.dir / "snapshots" / f"{stamp}_{re.sub(r'[^A-Za-z0-9_-]', '_', key)}.jpg")
            cv2.imwrite(snap, frame)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), f"{t:.1f}", severity, message, snap])
        if self.on_alert:
            self.on_alert(f"[{severity}] {message}")
        return True
