"""
alignment.py
------------
Matches hooks ↔ lugs and computes alignment status.

✔ Uses bottom-center hook anchor
✔ Uses lug bounding box
✔ Dynamic tolerance (real-world offset)
✔ 3-level alignment:
    - ALIGNED
    - NEARLY ALIGNED
    - NOT ALIGNED
✔ Outputs dx/dy/distance in cm
"""

import numpy as np


LUG_REAL_WIDTH_cm = 110.0


# ──────────────────────────────────────────────────────────────
def _bbox_center(det: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = det["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ──────────────────────────────────────────────────────────────
def _hook_anchor(det: dict, pos_ratio: float = 0.70) -> tuple[float, float]:
    """
    Bottom-center anchor of hook (slightly above bottom).
    """
    x1, y1, x2, y2 = det["bbox"]
    height = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = y1 + pos_ratio * (y2 - y1)   

    return (cx, cy)


# ──────────────────────────────────────────────────────────────
def _closest_point_on_bbox(point, bbox):
    px, py = point
    x1, y1, x2, y2 = bbox
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return (float(cx), float(cy))


# ──────────────────────────────────────────────────────────────
def _point_to_bbox_distance(point, bbox):
    closest_x, closest_y = _closest_point_on_bbox(point, bbox)
    px, py = point
    dx = float(px) - closest_x
    dy = float(py) - closest_y
    return dx, dy, float(np.hypot(dx, dy))


# ──────────────────────────────────────────────────────────────
def _pixel_scale_cm_per_px(lug: dict):
    x1, _, x2, _ = lug["bbox"]
    width_px = max(float(x2 - x1), 1.0)
    return float(LUG_REAL_WIDTH_cm / width_px)


# ──────────────────────────────────────────────────────────────
def match_hooks_lugs(hooks, lugs):
    if not hooks or not lugs:
        return []

    hook_points = np.array([_hook_anchor(h) for h in hooks])

    dist_mat = np.zeros((len(hooks), len(lugs)), dtype=float)

    for i, hook_point in enumerate(hook_points):
        for j, lug in enumerate(lugs):
            _, _, dist = _point_to_bbox_distance(tuple(hook_point), lug["bbox"])
            dist_mat[i, j] = dist

    pairs = []
    used_lug = set()

    for i, hook in enumerate(hooks):
        row = dist_mat[i].copy()

        for used_j in used_lug:
            row[used_j] = np.inf

        j = int(np.argmin(row))

        if row[j] < np.inf:
            pairs.append((hook, lugs[j]))
            used_lug.add(j)

    return pairs


# ──────────────────────────────────────────────────────────────
def compute_alignment(hook, lug, threshold=30):
    """
    Alignment logic with real-world tolerance.
    """

    # ── Anchor point ───────────────────────────────────
    tx, ty = _hook_anchor(hook)

    lx1, ly1, lx2, ly2 = lug["bbox"]

    # ── Scaling ────────────────────────────────────────
    cm_per_px = _pixel_scale_cm_per_px(lug)
    threshold_cm = threshold * cm_per_px

    # ── Distance from lug box ──────────────────────────
    dx_px, dy_px, dist_px = _point_to_bbox_distance((tx, ty), lug["bbox"])

    dx = dx_px * cm_per_px
    dy = dy_px * cm_per_px
    distance = dist_px * cm_per_px

    # ───────────────────────────────────────────────────
    # 🔥 Dynamic tolerance (KEY PART)
    # ───────────────────────────────────────────────────
    lug_width  = lx2 - lx1
    lug_height = ly2 - ly1

    offset_x = 0.25 * lug_width   # horizontal tolerance
    offset_y = 0.35 * lug_height  # vertical tolerance

    # ── Strict inside (perfect) ────────────────────────
    tight_inside = (
        lx1 - offset_x <= tx <= lx2 + offset_x and
        ly1 - offset_y <= ty <= ly2 + offset_y
    )

    # ── Loose inside (real-world acceptable) ───────────
    loose_inside = (
        lx1 - offset_x <= tx <= lx2 + offset_x and
        ly1 - offset_y <= ty <= ly2 + offset_y
    )

    # ───────────────────────────────────────────────────
    # 🎯 Alignment classification
    # ───────────────────────────────────────────────────
    if tight_inside:
        return {
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "distance": round(distance, 1),
            "units": "cm",
            "status": "ALIGNED",
            "direction": "Aligned",
        }

    if loose_inside:
        return {
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "distance": round(distance, 1),
            "units": "cm",
            "status": "NEARLY ALIGNED",
            "direction": "Fine adjust",
        }

    # ───────────────────────────────────────────────────
    # Direction logic (only when NOT aligned)
    # ───────────────────────────────────────────────────
    h_dir = ""
    v_dir = ""

    if dx < -threshold_cm:
        h_dir = "Move Right"
    elif dx > threshold_cm:
        h_dir = "Move Left"

    if dy < -threshold_cm:
        v_dir = "Move Down"
    elif dy > threshold_cm:
        v_dir = "Move Up"

    direction = " & ".join(filter(None, [h_dir, v_dir])) or "Adjust"

    return {
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "distance": round(distance, 1),
        "units": "cm",
        "status": "NOT ALIGNED",
        "direction": direction,
    }