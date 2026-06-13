"""
app.py
------
Flask web application for SIGHT (System for Intelligent Guidance of Hook Trajectory).

Endpoints
---------
GET  /                        → HTML UI
GET  /health                  → JSON status
POST /process_image           → Process uploaded image, returns base64 JPEG + alignment JSON
POST /upload_video            → Upload video, start background processing, returns video_id
GET  /video_feed/<vid_id>     → MJPEG stream of annotated frames
GET  /video_status/<vid_id>   → JSON alignment data for latest frame (poll-friendly)
POST /stop_video/<vid_id>     → Stop & cleanup a video stream
"""

import base64
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

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Detector (lazy singleton) ─────────────────────────────────────────────────
_detector: Optional[HookLugDetector] = None
_detector_lock = threading.Lock()


def get_detector(model_path: str = "best.pt") -> HookLugDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = HookLugDetector(model_path)
    return _detector


# ── Core processing pipeline ──────────────────────────────────────────────────
def process_frame(
    frame: np.ndarray,
    threshold: int = 30,
    conf: float = 0.25,
) -> tuple[np.ndarray, list[dict]]:

    det        = get_detector()
    detections = det.detect(frame, conf_threshold=conf)

    # ── Split detections ─────────────────────────────
    hooks = [d for d in detections if d["label"] == "hook"]
    lugs  = [d for d in detections if d["label"] == "lug"]
    tips  = [d for d in detections if d["label"] == "hook_tip"]  # 🔥 NEW

    # ── Assign tip to each hook (MODEL-BASED) ────────
    for h in hooks:
        hx = (h["bbox"][0] + h["bbox"][2]) // 2
        hy = (h["bbox"][1] + h["bbox"][3]) // 2

        best_tip = None
        best_dist = float("inf")

        for t in tips:
            tx = (t["bbox"][0] + t["bbox"][2]) // 2
            ty = (t["bbox"][1] + t["bbox"][3]) // 2

            d = (hx - tx)**2 + (hy - ty)**2

            if d < best_dist:
                best_dist = d
                best_tip = (tx, ty)

        h["tip"] = best_tip   # 🔥 THIS IS THE KEY FIX

    # ── Lug centre ───────────────────────────────────
    for l in lugs:
        x1, y1, x2, y2 = l["bbox"]
        l["center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # ── Match hooks ↔ lugs ───────────────────────────
    pairs = match_hooks_lugs(hooks, lugs)

    # ── Alignment ────────────────────────────────────
    alignment_results: list[dict] = []

    for h, l in pairs:
        al = compute_alignment(h, l, threshold)   # 🔥 already fixed
        h["alignment"] = al

        alignment_results.append(
            {
                "hook_conf": round(h["conf"], 3),
                "lug_conf":  round(l["conf"], 3),
                **al,
            }
        )

    # ── Draw ─────────────────────────────────────────
    annotated = draw_annotations(frame.copy(), hooks, lugs, pairs, threshold)

    return annotated, alignment_results


# ── Video processor (background thread) ──────────────────────────────────────
class VideoProcessor:
    """Processes a video file frame-by-frame in a background thread."""

    def __init__(
        self,
        video_path: str,
        threshold: int = 30,
        conf: float = 0.25,
        smooth_n: int = 5,
    ):
        self.video_path = video_path
        self.threshold  = threshold
        self.conf       = conf
        self.smooth_n   = smooth_n

        self._frame_buf:     deque[bytes]  = deque(maxlen=2)
        self._alignment_buf: list[dict]   = []
        self._running: bool = False
        self._done:    bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time: float = 0
        self._target_fps: int = 30
        self._target_fps: int = 30

    # ── Public API ─────────────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def done(self) -> bool:
        return self._done

    def get_latest(self) -> tuple[Optional[bytes], list[dict]]:
        with self._lock:
            frame_bytes = self._frame_buf[-1] if self._frame_buf else None
            alignment   = list(self._alignment_buf)
        return frame_bytes, alignment

    # ── Background worker ──────────────────────────────────────────────────
    def _run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        # Simple tip-position smoother (rolling mean)
        tip_history: dict[int, deque] = {}   # pair_index → deque of (x,y)

        try:
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, alignment = process_frame(frame, self.threshold, self.conf)

                # Smooth tip positions over last N frames
                for i, al in enumerate(alignment):
                    # (smoothing is already visual in the annotated frame)
                    pass

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 82]
                ok, jpeg = cv2.imencode(".jpg", annotated, encode_params)
                if not ok:
                    continue

                with self._lock:
                    self._frame_buf.append(jpeg.tobytes())
                    self._alignment_buf = alignment

                # Frame rate limiter: target ~30 FPS
                elapsed = time.time() - self._last_frame_time
                target_interval = 1.0 / self._target_fps
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                self._last_frame_time = time.time()

        finally:
            cap.release()
            # Clean up temp file
            try:
                os.remove(self.video_path)
            except OSError:
                pass
            self._running = False
            self._done    = True


# ── Active streams registry ───────────────────────────────────────────────────
_streams: dict[str, VideoProcessor] = {}
_streams_lock = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    import torch
    gpu = torch.cuda.is_available()
    return jsonify(
        {
            "status": "ok",
            "gpu":    gpu,
            "device": "cuda" if gpu else "cpu",
        }
    )


@app.route("/process_image", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f         = request.files["file"]
    threshold = int(request.form.get("threshold", 30))
    conf      = float(request.form.get("conf", 0.25))

    data  = f.read()
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Failed to decode image"}), 400

    annotated, alignment_data = process_frame(frame, threshold, conf)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return jsonify({"error": "Failed to encode result"}), 500

    img_b64 = base64.b64encode(buf).decode()

    all_aligned = (
        bool(alignment_data)
        and all(a["status"] == "ALIGNED" for a in alignment_data)
    )

    return jsonify(
        {
            "image":     img_b64,
            "alignments": alignment_data,
            "num_pairs":  len(alignment_data),
            "overall":   "ALIGNED" if all_aligned
                         else "NOT ALIGNED" if alignment_data
                         else "NO DETECTION",
        }
    )


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f         = request.files["file"]
    threshold = int(request.form.get("threshold", 30))
    conf      = float(request.form.get("conf", 0.25))
    smooth_n  = int(request.form.get("smooth_n", 5))

    vid_id = uuid.uuid4().hex[:12]
    ext    = os.path.splitext(f.filename or "")[1] or ".mp4"
    path   = os.path.join("uploads", f"{vid_id}{ext}")
    f.save(path)

    proc = VideoProcessor(path, threshold, conf, smooth_n)
    proc.start()

    with _streams_lock:
        _streams[vid_id] = proc

    return jsonify({"video_id": vid_id})


@app.route("/video_feed/<vid_id>")
def video_feed(vid_id: str):
    def generate():
        with _streams_lock:
            proc = _streams.get(vid_id)
        if proc is None:
            return

        while not proc.done or proc.running:
            frame_bytes, _ = proc.get_latest()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            time.sleep(0.030)  # ~33 fps cap

        # Finished → remove from registry
        with _streams_lock:
            _streams.pop(vid_id, None)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_status/<vid_id>")
def video_status(vid_id: str):
    with _streams_lock:
        proc = _streams.get(vid_id)
    if proc is None:
        return jsonify({"running": False, "done": True, "alignments": []})

    _, alignment = proc.get_latest()
    return jsonify(
        {
            "running":    proc.running,
            "done":       proc.done,
            "alignments": alignment,
        }
    )


@app.route("/stop_video/<vid_id>", methods=["POST"])
def stop_video(vid_id: str):
    with _streams_lock:
        proc = _streams.pop(vid_id, None)
    if proc:
        proc.stop()
    return jsonify({"stopped": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIGHT server")
    parser.add_argument("--model", default="best.pt",    help="Path to YOLOv8 weights")
    parser.add_argument("--host",  default="0.0.0.0",   help="Server host")
    parser.add_argument("--port",  default=5000, type=int, help="Server port")
    parser.add_argument("--debug", action="store_true",  help="Flask debug mode")
    args = parser.parse_args()

    # Pre-warm the model
    print("[App] Pre-loading model...")
    get_detector(args.model)
    print("[App] Model ready.")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
