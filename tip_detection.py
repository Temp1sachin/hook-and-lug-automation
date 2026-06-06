"""
tip_detection.py
----------------
Hybrid hook tip detection:

Priority:
1. Use YOLO-detected hook_tip (best)
2. Fallback to geometric method (robust)
"""

import cv2
import numpy as np
from typing import Optional, List


# ──────────────────────────────────────────────────────────────
def detect_hook_tip(
    frame: np.ndarray,
    hook_bbox: list[int],
    detections: Optional[List[dict]] = None,   # 🔥 NEW
) -> Optional[tuple[int, int]]:
    """
    Detect hook tip using:
    1. Model detection (preferred)
    2. Fallback geometric method
    """

    x1, y1, x2, y2 = [int(v) for v in hook_bbox]

    # ──────────────────────────────────────────────────────────
    # ✅ STEP 1: USE MODEL DETECTIONS (BEST)
    # ──────────────────────────────────────────────────────────
    if detections is not None:
        tips = [d for d in detections if d["class"] == 2]  # hook_tip

        if tips:
            # pick nearest tip to this hook
            hx = (x1 + x2) // 2
            hy = (y1 + y2) // 2

            best_tip = None
            best_dist = float("inf")

            for t in tips:
                tx = (t["bbox"][0] + t["bbox"][2]) // 2
                ty = (t["bbox"][1] + t["bbox"][3]) // 2

                d = (hx - tx)**2 + (hy - ty)**2

                if d < best_dist:
                    best_dist = d
                    best_tip = (tx, ty)

            if best_tip:
                return best_tip

    # ──────────────────────────────────────────────────────────
    # ⚠️ STEP 2: FALLBACK (simple + stable)
    # ──────────────────────────────────────────────────────────

    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    edges = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return (x1 + (x2 - x1)//2, y2)

    largest = max(contours, key=cv2.contourArea)
    pts = largest[:, 0, :].astype(float)

    if len(pts) < 10:
        return (x1 + (x2 - x1)//2, y2)

    # 🔥 SIMPLE + STABLE: farthest point from centroid
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    dist = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
    idx = np.argmax(dist)

    local_x = int(pts[idx, 0])
    local_y = int(pts[idx, 1])

    tip_x = x1 + local_x
    tip_y = y1 + local_y

    return (tip_x, tip_y)


# ──────────────────────────────────────────────────────────────
# DEBUG
# ──────────────────────────────────────────────────────────────
def get_edge_debug_image(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]

    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    crop = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 100)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)