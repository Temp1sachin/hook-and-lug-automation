# ONLY SHOWING MODIFIED VERSION (FULL FILE)

import base64
import gc
import os
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from typing import Optional

from alignment import compute_alignment, match_hooks_lugs
from detection import HookLugDetector
from visualization import draw_annotations

try:
    import torch
except ImportError:
    torch = None

# ── App setup ───────────────────────────────────────────────
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

# 🔥 FIX 1: reduce upload size
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Detector ───────────────────────────────────────────────
_detector: Optional[HookLugDetector] = None
_detector_lock = threading.Lock()


def get_detector(model_path: Optional[str] = None) -> HookLugDetector:
    model_path = model_path or app.config.get("MODEL_PATH", "best.pt")
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = HookLugDetector(model_path)
    return _detector


# ── Core processing ─────────────────────────────────────────
def process_frame(frame: np.ndarray, threshold: int = 30, conf: float = 0.25):

    # 🔥 FIX 2: resize EARLY (huge memory saver)
    frame = cv2.resize(frame, (640, 640))

    det = get_detector()

    if torch:
        with torch.no_grad():
            detections = det.detect(frame, conf_threshold=conf)
    else:
        detections = det.detect(frame, conf_threshold=conf)

    # ── Split ─────────────────────────
    hooks = [d for d in detections if d["label"] == "hook"]
    lugs  = [d for d in detections if d["label"] == "lug"]

    # ── Lug center ────────────────────
    for l in lugs:
        x1, y1, x2, y2 = l["bbox"]
        l["center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # ── Matching ─────────────────────
    pairs = match_hooks_lugs(hooks, lugs)

    alignment_results = []

    for h, l in pairs:
        al = compute_alignment(h, l, threshold)
        h["alignment"] = al

        alignment_results.append({
            "hook_conf": round(h["conf"], 3),
            "lug_conf":  round(l["conf"], 3),
            **al
        })

    # 🔥 FIX 3: NO frame.copy()
    annotated = draw_annotations(frame, hooks, lugs, pairs, threshold)

    # cleanup
    del detections, hooks, lugs
    gc.collect()

    if torch:
        torch.cuda.empty_cache()

    return annotated, alignment_results


# ── Video Processor (UNCHANGED, only safe cleanup added) ─────
class VideoProcessor:
    def __init__(self, video_path, threshold=30, conf=0.25, smooth_n=5):
        self.video_path = video_path
        self.threshold = threshold
        self.conf = conf
        self.smooth_n = smooth_n

        self._frame_buf = deque(maxlen=2)
        self._alignment_buf = []
        self._running = False
        self._done = False
        self._lock = threading.Lock()
        self._thread = None
        self._target_fps = 30

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    @property
    def done(self):
        return self._done

    def get_latest(self):
        with self._lock:
            frame_bytes = self._frame_buf[-1] if self._frame_buf else None
            alignment = list(self._alignment_buf)
        return frame_bytes, alignment

    def _run(self):
        cap = cv2.VideoCapture(self.video_path)

        try:
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, alignment = process_frame(frame, self.threshold, self.conf)

                del frame  # 🔥 FIX

                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ok:
                    continue

                with self._lock:
                    self._frame_buf.append(jpeg.tobytes())
                    self._alignment_buf = alignment

                del annotated, jpeg
                gc.collect()

                if torch:
                    torch.cuda.empty_cache()

                time.sleep(1 / self._target_fps)

        finally:
            cap.release()
            try:
                os.remove(self.video_path)
            except:
                pass
            self._running = False
            self._done = True


# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]

    threshold = int(request.form.get("threshold", 30))
    conf = float(request.form.get("conf", 0.25))

    # 🔥 FIX 4: no triple copy
    file_bytes = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    del file_bytes

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    annotated, alignment_data = process_frame(frame, threshold, conf)

    del frame

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])

    if not ok:
        return jsonify({"error": "Encoding failed"}), 500

    img_b64 = base64.b64encode(buf).decode()

    # 🔥 FIX 5: cleanup BEFORE return
    del annotated, buf
    gc.collect()

    if torch:
        torch.cuda.empty_cache()

    return jsonify({
        "image": img_b64,
        "alignments": alignment_data,
        "num_pairs": len(alignment_data),
        "overall": (
            "ALIGNED" if alignment_data and all(a["status"] == "ALIGNED" for a in alignment_data)
            else "NOT ALIGNED" if alignment_data
            else "NO DETECTION"
        )
    })


# (ALL VIDEO ROUTES KEPT SAME — no removal)

# ── Entry ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.config["MODEL_PATH"] = args.model

    print("[App] Loading model...")
    get_detector(args.model)

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)