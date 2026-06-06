"""
alignment.py
------------
Matches hooks ↔ lugs and computes alignment status.

Uses:
✔ hook_tip (from model)
✔ lug center
✔ robust pairing + safe checks
"""

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────
def _bbox_center(det: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = det["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ──────────────────────────────────────────────────────────────
def match_hooks_lugs(
    hooks: list[dict], lugs: list[dict]
) -> list[tuple[dict, dict]]:
    """
    Greedy nearest-neighbour matching between hooks and lugs.
    """

    if not hooks or not lugs:
        return []

    hook_centers = np.array([_bbox_center(h) for h in hooks])
    lug_centers  = np.array([_bbox_center(l) for l in lugs])

    diff     = hook_centers[:, None, :] - lug_centers[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))

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
def compute_alignment(
    hook: dict,
    lug: dict,
    threshold: int = 30,
) -> dict:
    """
    Compute alignment using hook_tip and lug center.
    """

    hook_tip = hook.get("tip")   # 🔥 from model
    lug_center = _bbox_center(lug)

    # ── Safety check ─────────────────────────────────────
    if hook_tip is None:
        return {
            "dx": None,
            "dy": None,
            "distance": None,
            "status": "UNKNOWN",
            "direction": "Tip missing",
        }

    dx = float(hook_tip[0]) - float(lug_center[0])
    dy = float(hook_tip[1]) - float(lug_center[1])
    distance = float(np.hypot(dx, dy))

    # ── Alignment check ──────────────────────────────────
    if abs(dx) < threshold and abs(dy) < threshold:
        return {
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "distance": round(distance, 1),
            "status": "ALIGNED",
            "direction": "Aligned",
        }

    # ── Direction logic ──────────────────────────────────
    h_dir = ""
    v_dir = ""

    if dx < -threshold:
        h_dir = "Move Right"
    elif dx > threshold:
        h_dir = "Move Left"

    if dy < -threshold:
        v_dir = "Move Down"
    elif dy > threshold:
        v_dir = "Move Up"

    direction = " & ".join(filter(None, [h_dir, v_dir])) or "Adjust"

    return {
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "distance": round(distance, 1),
        "status": "NOT ALIGNED",
        "direction": direction,
    }