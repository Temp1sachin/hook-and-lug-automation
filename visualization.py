"""
visualization.py
----------------
Draws all detection annotations onto an OpenCV frame.

Colour legend
  Hook bounding box  : amber / orange
  Lug  bounding box  : steel blue
  Hook tip dot       : red
  Lug  centre dot    : cyan
  Connecting line    : green (aligned) | red (not aligned)
  Status badge       : green bg (aligned) | red bg (not aligned) | grey (unknown)
"""

import cv2
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
_HOOK_BOX   = (0,  165, 255)   # BGR amber-orange
_LUG_BOX    = (200, 130,  20)  # BGR steel-blue
_HOOK_TIP   = (0,   0,  220)   # BGR red
_LUG_CTR    = (220, 180,   0)  # BGR cyan
_LINE_OK    = (40,  200,  40)   # BGR green
_LINE_BAD   = (40,   40, 200)  # BGR red
_TEXT_OK    = (60,  230,  60)
_TEXT_BAD   = (60,   60, 230)
_BG_OK      = (20,  100,  20)
_BG_BAD     = (20,   20, 140)
_BG_UNK     = (50,   50,  50)
_WHITE      = (255, 255, 255)
_BLACK      = (  0,   0,   0)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MONO  = cv2.FONT_HERSHEY_DUPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_box(img, x1, y1, x2, y2, color, text, thickness=2):
    """Draw a coloured bounding box with a filled label tag."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        (tw, th), bl = cv2.getTextSize(text, _FONT, 0.48, 1)
        label_y = max(y1 - 2, th + 4)
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 2), color, -1)
        cv2.putText(img, text, (x1 + 4, label_y - 1), _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)


def _dot(img, cx, cy, color, r=7):
    """Draw a filled dot with a white outline."""
    cx, cy = int(cx), int(cy)
    cv2.circle(img, (cx, cy), r,     color,  -1)
    cv2.circle(img, (cx, cy), r + 2, _WHITE,  1)


def _badge(img, x, y, text, bg):
    """Draw a filled status badge."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.55, 2)
    pad = 6
    cv2.rectangle(img, (x, y), (x + tw + 2*pad, y + th + 2*pad), bg,     -1)
    cv2.rectangle(img, (x, y), (x + tw + 2*pad, y + th + 2*pad), _WHITE,  1)
    cv2.putText(img, text, (x + pad, y + th + pad), _FONT, 0.55, _WHITE, 2, cv2.LINE_AA)
    return x + tw + 2*pad, y + th + 2*pad   # bottom-right corner


def _midpoint_label(img, p1, p2, text, color):
    """Draw a label at the midpoint of a line."""
    mx = int((p1[0] + p2[0]) / 2)
    my = int((p1[1] + p2[1]) / 2)
    (tw, th), _ = cv2.getTextSize(text, _FONT_MONO, 0.42, 1)
    cv2.rectangle(img, (mx - 3, my - th - 4), (mx + tw + 3, my + 2), _BLACK, -1)
    cv2.putText(img, text, (mx, my - 2), _FONT_MONO, 0.42, color, 1, cv2.LINE_AA)


# ── Main annotation function ──────────────────────────────────────────────────

def draw_annotations(
    frame:     np.ndarray,
    hooks:     list[dict],
    lugs:      list[dict],
    pairs:     list[tuple[dict, dict]],
    threshold: int = 30,
) -> np.ndarray:
    """
    Render all detection annotations.

    Parameters
    ----------
    frame     : BGR frame (will be copied internally)
    hooks     : list of hook detection dicts (from detector + tip_detection)
    lugs      : list of lug detection dicts (from detector)
    pairs     : list of (hook, lug) matched pairs
    threshold : alignment pixel threshold (for badge colour logic)

    Returns
    -------
    Annotated BGR frame.
    """
    img = frame.copy()

    paired_hook_ids = {id(h) for h, _ in pairs}
    paired_lug_ids  = {id(l) for _, l in pairs}

    # ── Draw all lug boxes ─────────────────────────────────────────────────
    for lug in lugs:
        x1, y1, x2, y2 = lug["bbox"]
        cx, cy = lug["center"]
        _label_box(img, x1, y1, x2, y2, _LUG_BOX, f"Lug {lug['conf']:.2f}")
        _dot(img, cx, cy, _LUG_CTR)

    # ── Draw all paired hook-lug annotations ───────────────────────────────
    for hook, lug in pairs:
        hx1, hy1, hx2, hy2 = hook["bbox"]
        tip     = hook.get("tip")
        center  = lug.get("center")
        al      = hook.get("alignment", {})
        status  = al.get("status", "UNKNOWN")

        is_aligned  = status == "ALIGNED"
        line_color  = _LINE_OK   if is_aligned else _LINE_BAD
        text_color  = _TEXT_OK   if is_aligned else _TEXT_BAD
        bg_color    = _BG_OK     if is_aligned else _BG_BAD

        # Hook bounding box
        _label_box(img, hx1, hy1, hx2, hy2, _HOOK_BOX, f"Hook {hook['conf']:.2f}")

        # Connecting line + dx/dy label
        if tip and center:
            cv2.line(
                img,
                (int(tip[0]),    int(tip[1])),
                (int(center[0]), int(center[1])),
                line_color, 2, cv2.LINE_AA,
            )
            dx, dy = al.get("dx"), al.get("dy")
            if dx is not None:
                _midpoint_label(
                    img, tip, center,
                    f"dx={dx:+.0f}  dy={dy:+.0f}",
                    text_color,
                )

        # Hook tip (red dot)
        if tip:
            _dot(img, tip[0], tip[1], _HOOK_TIP)

        # Lug centre (cyan dot) — re-draw on top of line
        if center:
            _dot(img, center[0], center[1], _LUG_CTR)

        # Status badge below hook box
        direction = al.get("direction", status)
        badge_x   = hx1
        badge_y   = hy2 + 6
        _badge(img, badge_x, badge_y, direction, bg_color)

    # ── Unpaired hooks (no matching lug) ───────────────────────────────────
    for hook in hooks:
        if id(hook) in paired_hook_ids:
            continue
        hx1, hy1, hx2, hy2 = hook["bbox"]
        tip = hook.get("tip")
        _label_box(img, hx1, hy1, hx2, hy2, _HOOK_BOX, f"Hook {hook['conf']:.2f}")
        if tip:
            _dot(img, tip[0], tip[1], _HOOK_TIP)
        _badge(img, hx1, hy2 + 6, "No Lug Matched", _BG_UNK)

    # ── Unpaired lugs (no matching hook) ───────────────────────────────────
    # Already drawn above; nothing extra needed.

    return img
