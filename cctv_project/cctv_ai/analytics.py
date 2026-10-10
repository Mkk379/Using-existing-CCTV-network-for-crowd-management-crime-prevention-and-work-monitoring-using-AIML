"""Crowd levels, density heatmap and zone-based (work / restricted) monitoring.
All anonymous: only counts and positions are used, no identities."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

LEVEL_COLORS = {"LOW": (0, 200, 0), "MEDIUM": (0, 215, 255), "HIGH": (0, 140, 255), "CRITICAL": (0, 0, 255)}


def crowd_level(count, s):
    if count >= s.crowd_critical:
        return "CRITICAL"
    if count >= s.crowd_high:
        return "HIGH"
    if count >= s.crowd_medium:
        return "MEDIUM"
    return "LOW"


def box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def box_foot(b):
    return ((b[0] + b[2]) / 2, b[3])


class Heatmap:
    """Accumulates where people stand over time."""

    def __init__(self, shape, decay=0.9995):
        self.shape = shape
        self.acc = np.zeros(shape, np.float32)
        self.decay = decay

    def add(self, points, radius=18):
        mask = np.zeros_like(self.acc)
        for x, y in points:
            cv2.circle(mask, (int(x), int(y)), radius, 1.0, -1)
        self.acc = self.acc * self.decay + mask

    def overlay(self, frame, alpha=0.45):
        if self.acc.max() <= 0:
            return frame
        blur = cv2.GaussianBlur(self.acc, (0, 0), 15)
        norm = blur / max(blur.max(), 1e-6)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1 - alpha, heat, alpha, 0)
        return np.where((norm > 0.08)[..., None], blended, frame)


@dataclass
class Zone:
    name: str
    kind: str            # "work" (track presence/idle) or "restricted" (alert on entry)
    x1: float            # coordinates are normalised 0..1 so they work at any resolution
    y1: float
    x2: float
    y2: float

    def contains(self, px, py, w, h):
        return self.x1 * w <= px <= self.x2 * w and self.y1 * h <= py <= self.y2 * h

    def pixel_rect(self, w, h):
        return int(self.x1 * w), int(self.y1 * h), int(self.x2 * w), int(self.y2 * h)


def load_zones(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        return [Zone(**d) for d in json.loads(p.read_text())]
    except Exception:
        return []


def save_zones(path, zones):
    Path(path).write_text(json.dumps([asdict(z) for z in zones], indent=2))


class ZoneMonitor:
    def __init__(self, zones, absent_alert_sec):
        self.zones = zones
        self.absent_alert_sec = absent_alert_sec
        self.state = {z.name: dict(present=None, count=0, last_t=None, present_sec=0.0,
                                   absent_sec=0.0, absent_since=None, alerted=False, breaches=0)
                      for z in zones}

    def update(self, centers, t, w, h):
        """Returns list of (key, severity, message) events."""
        events = []
        for z in self.zones:
            st = self.state[z.name]
            inside = sum(1 for (cx, cy) in centers if z.contains(cx, cy, w, h))
            present = inside > 0
            if z.kind == "work":
                if st["last_t"] is not None:
                    dt = min(t - st["last_t"], 5.0)
                    st["present_sec" if st["present"] else "absent_sec"] += max(dt, 0)
                if present:
                    st["absent_since"], st["alerted"] = None, False
                else:
                    if st["absent_since"] is None:
                        st["absent_since"] = t
                    if not st["alerted"] and t - st["absent_since"] >= self.absent_alert_sec:
                        st["alerted"] = True
                        events.append((f"absent_{z.name}", "MEDIUM",
                                       f"Work zone '{z.name}' empty for {int(t - st['absent_since'])}s"))
            else:  # restricted
                if present:
                    if not st["present"]:
                        st["breaches"] += 1
                    events.append((f"intrusion_{z.name}", "HIGH",
                                   f"Restricted zone '{z.name}': {inside} person(s) detected"))
            st["present"], st["count"], st["last_t"] = present, inside, t
        return events

    def summary(self):
        rows = []
        for z in self.zones:
            st = self.state[z.name]
            if z.kind == "work":
                rows.append(f"Work zone '{z.name}': present {st['present_sec']:.0f}s, away {st['absent_sec']:.0f}s")
            else:
                rows.append(f"Restricted zone '{z.name}': {st['breaches']} entries")
        return rows
