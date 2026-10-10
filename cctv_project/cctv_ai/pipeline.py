"""Frame analyzer (detect + analytics + drawing) and a threaded stream processor."""
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .alerts import AlertManager
from .analytics import (LEVEL_COLORS, Heatmap, ZoneMonitor, box_center, box_foot, crowd_level)

GREEN, RED = (0, 255, 0), (0, 0, 255)
ZONE_COLORS = {"work": (255, 128, 0), "restricted": (0, 128, 255)}


def fit_width(frame, max_w):
    h, w = frame.shape[:2]
    if w > max_w:
        s = max_w / w
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    return frame


class Analyzer:
    def __init__(self, settings, person_det, weapon_det, zones, on_alert=None):
        self.s = settings
        self.person_det, self.weapon_det = person_det, weapon_det
        self.zones = zones
        self.monitor = ZoneMonitor(zones, settings.absent_alert_sec)
        self.alerts = AlertManager(settings.output_dir, settings.alert_cooldown_sec, on_alert)
        self.heat = None
        self.persons, self.weapons = [], []
        self.level, self.fps = "LOW", 0.0
        self.peak, self.total, self.n, self.weapon_frames = 0, 0, 0, 0

    # ------------------------------------------------------------------
    def analyze(self, frame, t, detect=True):
        h, w = frame.shape[:2]
        if self.heat is None or self.heat.shape != (h, w):
            self.heat = Heatmap((h, w))
        events = []
        if detect:
            self.persons = self.person_det.detect(frame) if self.person_det else []
            self.weapons = self.weapon_det.detect(frame) if self.weapon_det else []
            count = len(self.persons)
            self.level = crowd_level(count, self.s)
            self.heat.add([box_foot(b) for b in self.persons])
            events += self.monitor.update([box_center(b) for b in self.persons], t, w, h)
            if self.level in ("HIGH", "CRITICAL"):
                events.append((f"crowd_{self.level}", self.level, f"Crowd level {self.level}: {count} people"))
            for wb in self.weapons:
                events.append((f"weapon_{wb[5]}", "CRITICAL", f"Possible weapon detected: {wb[5]} ({wb[4]:.0%})"))
            self.peak = max(self.peak, count)
            self.total += count
            self.n += 1
            self.weapon_frames += 1 if self.weapons else 0

        out = self._draw(frame)
        for key, sev, msg in events:
            self.alerts.raise_alert(key, sev, msg, t, out)
        return out, self.stats()

    # ------------------------------------------------------------------
    def stats(self):
        return dict(people=len(self.persons), level=self.level, weapons=[b[5] for b in self.weapons],
                    peak=self.peak, avg=(self.total / self.n if self.n else 0.0),
                    zones=[(z.name, z.kind, self.monitor.state[z.name]["count"]) for z in self.zones],
                    alerts=self.alerts.count)

    def summary(self):
        lines = [f"Peak people: {self.peak}, average: {self.total / self.n if self.n else 0:.1f}",
                 f"Frames with weapon detections: {self.weapon_frames}",
                 f"Alerts raised: {self.alerts.count} (see {self.alerts.csv_path})"]
        return lines + self.monitor.summary()

    # ------------------------------------------------------------------
    def _draw(self, frame):
        out = self.heat.overlay(frame) if self.s.show_heatmap else frame.copy()
        h, w = out.shape[:2]
        for z in self.zones:
            x1, y1, x2, y2 = z.pixel_rect(w, h)
            col = ZONE_COLORS.get(z.kind, (255, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            n = self.monitor.state[z.name]["count"]
            cv2.putText(out, f"{z.name} ({z.kind}): {n}", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for x1, y1, x2, y2, _ in self.persons:
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
        for x1, y1, x2, y2, c, label in self.weapons:
            cv2.rectangle(out, (x1, y1), (x2, y2), RED, 3)
            cv2.putText(out, f"{label} {c:.0%}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        # HUD
        lc = LEVEL_COLORS[self.level]
        lines = [(f"People: {len(self.persons)}", GREEN), (f"Crowd: {self.level}", lc),
                 (f"Weapons: {len(self.weapons)}", RED if self.weapons else (255, 255, 255))]
        if self.fps:
            lines.append((f"FPS: {self.fps:.1f}", (255, 255, 255)))
        cv2.rectangle(out, (0, 0), (190, 12 + 26 * len(lines)), (0, 0, 0), -1)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return out


def analyze_image(path, analyzer, max_w=1280):
    frame = cv2.imread(path)
    if frame is None:
        return None, None
    return analyzer.analyze(fit_width(frame, max_w), 0.0)


class StreamProcessor(threading.Thread):
    """Runs capture + analysis in a background thread so the GUI stays responsive."""

    def __init__(self, source, analyzer, settings, on_frame, on_stats, on_log, on_done):
        super().__init__(daemon=True)
        self.source, self.a, self.s = source, analyzer, settings
        self.on_frame, self.on_stats, self.on_log, self.on_done = on_frame, on_stats, on_log, on_done
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        src = self.source
        is_file = isinstance(src, str) and Path(src).exists()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.on_log(f"Failed to open source: {src}")
            self.on_done()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 < (fps or 0) <= 120 else 25.0
        out_dir = Path(self.s.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "crowd_log.csv"
        new_log = not log_path.exists()
        logf = open(log_path, "a", newline="")
        log = csv.writer(logf)
        if new_log:
            log.writerow(["timestamp", "source", "video_time_s", "people", "level", "weapons"])

        start = time.time()
        prev, ema, idx, fails, last_sec = start, 0.0, 0, 0, -1
        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        break
                    fails += 1
                    if fails > 5:
                        self.on_log("Stream lost; giving up after 5 reconnect attempts.")
                        break
                    self.on_log(f"Stream read failed, reconnecting ({fails}/5)...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(src)
                    continue
                fails = 0
                frame = fit_width(frame, self.s.max_process_width)
                t = idx / fps if is_file else time.time() - start
                detect = idx % max(1, self.s.detect_every_n_frames) == 0
                out, stats = self.a.analyze(frame, t, detect)
                now = time.time()
                ema = 0.9 * ema + 0.1 / max(now - prev, 1e-3) if ema else 1 / max(now - prev, 1e-3)
                prev, self.a.fps = now, ema
                self.on_frame(out)
                self.on_stats(stats)
                if int(t) != last_sec:
                    last_sec = int(t)
                    log.writerow([datetime.now().isoformat(timespec="seconds"), str(src), f"{t:.0f}",
                                  stats["people"], stats["level"], len(stats["weapons"])])
                if is_file:  # play at natural speed
                    wait = start + idx / fps - time.time()
                    if wait > 0:
                        time.sleep(min(wait, 0.1))
                idx += 1
        finally:
            cap.release()
            logf.close()
            self.on_log("--- Session summary ---")
            for line in self.a.summary():
                self.on_log(line)
            self.on_done()
