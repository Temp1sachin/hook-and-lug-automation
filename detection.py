"""
detection.py
------------
YOLO-based detector for hook, lug, and hook_tip.

Classes:
  0 → hook
  1 → lug
  2 → hook_tip
"""

import torch
from ultralytics import YOLO
import numpy as np


# 🔥 UPDATED CLASSES
CLASS_NAMES = {
    0: "hook",
    1: "lug",
    2: "hook_tip"
}


class HookLugDetector:
    """Wraps a YOLOv8 model for hook/lug/tip detection."""

    def __init__(self, model_path: str = "best.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Loading '{model_path}' on {self.device.upper()}")
        self.model = YOLO(model_path)
        self.model_path = model_path

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        """
        Run inference on a single BGR frame.
        """
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            device=self.device,
            conf=conf_threshold,
            verbose=False
        )

        detections: list[dict] = []

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Clamp
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                # Normalised
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)

                detections.append({
                    "class": cls,
                    "label": CLASS_NAMES.get(cls, f"class_{cls}"),
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_norm": [cx, cy, bw, bh],
                })

        return detections

    # ------------------------------------------------------------------
    def split_detections(self, detections: list[dict]):
        """
        🔥 NEW: Split detections into hooks, lugs, tips
        """
        hooks = [d for d in detections if d["class"] == 0]
        lugs  = [d for d in detections if d["class"] == 1]
        tips  = [d for d in detections if d["class"] == 2]

        return hooks, lugs, tips