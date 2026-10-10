"""Model wrappers. Both expose detect(frame_bgr) with simple tuple outputs."""
from pathlib import Path


class PersonDetector:
    """Person detector. Tries, in order:
    1. Ultralytics-format weights (e.g. yolov5su.pt, yolov8n.pt)
    2. Original YOLOv5 weights (your yolov5s.pt) through torch.hub
    3. Downloaded yolov8n.pt as a last resort
    """

    def __init__(self, model_path: str, conf: float = 0.35):
        self.conf = conf
        self.model = None
        self.backend = None
        errors = []
        path = model_path if Path(model_path).exists() else "yolov5su.pt"

        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.backend = f"ultralytics ({path})"
        except Exception as e:
            errors.append(f"ultralytics: {e}")

        if self.model is None:
            try:
                import torch
                self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
                self.model.conf = conf
                self.backend = f"yolov5 torch.hub ({model_path})"
            except Exception as e:
                errors.append(f"torch.hub: {e}")

        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self.backend = "ultralytics (yolov8n.pt fallback)"
            except Exception as e:
                errors.append(f"fallback: {e}")
                raise RuntimeError("Could not load any person model:\n" + "\n".join(errors))

    def detect(self, frame_bgr):
        """Return list of (x1, y1, x2, y2, conf) for people."""
        out = []
        if self.backend.startswith("yolov5 torch.hub"):
            res = self.model(frame_bgr[:, :, ::-1].copy())
            for x1, y1, x2, y2, c, cls in res.xyxy[0].cpu().numpy():
                if int(cls) == 0 and c >= self.conf:
                    out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        else:
            res = self.model.predict(frame_bgr, classes=[0], conf=self.conf, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0])))
        return out


class WeaponDetector:
    """Custom weapon model (any Ultralytics-compatible .pt). Labels come from the model itself."""

    def __init__(self, model_path: str, conf: float = 0.30):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Weapon model '{model_path}' not found. Put your trained best.pt next to app.py.")
        from ultralytics import YOLO
        self.conf = conf
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, frame_bgr):
        """Return list of (x1, y1, x2, y2, conf, label)."""
        out = []
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0]), str(self.names[int(b.cls[0])])))
        return out
