# S.I.G.H.T.

<strong>S</strong>ystem for <strong>I</strong>ntelligent <strong>G</strong>uidance of <strong>H</strong>ook <strong>T</strong>rajectory

A production-ready web application for detecting and verifying the alignment between **hooks** and **lugs** using a hybrid **YOLOv8 + OpenCV** pipeline.

---

## Architecture

```
hook_lug_app/
├── app.py              ← Flask web server + REST endpoints
├── detection.py        ← YOLOv8 inference wrapper
├── (tip_detection.py)  ← Deprecated: now we use synthetic hook anchor
├── alignment.py        ← Hook↔Lug matching + dx/dy alignment logic
├── visualization.py    ← OpenCV annotation drawing utilities
├── requirements.txt
├── best.pt             ← Your YOLOv8 weights (place here)
├── uploads/            ← Temporary upload directory (auto-created)
├── templates/
│   └── index.html      ← Main UI
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Detection Pipeline

```
Input Frame
    │
    ▼
[detection.py]  YOLOv8 inference
    │           → hook boxes (class 0)
    │           → lug  boxes (class 1)
    │
    ├──── For each HOOK ────────────────────────────────────────
    │     Synthetic anchor (stable):
    │       • Compute bbox midpoint and shift slightly downward
    │       • This deterministic anchor replaces tip-detection models
    │
    ├──── For each LUG ─────────────────────────────────────────
    │     lug_center = bbox centre (cx, cy)
    │
    ├──── [alignment.py]  Greedy nearest-neighbour matching ────
    │     Hook i → nearest unmatched Lug j
    │
    ├──── Per pair: compute dx, dy ─────────────────────────────
    │     dx = hook_anchor_x - lug_center_x
    │     dy = hook_anchor_y - lug_center_y
    │     if |dx| < threshold AND |dy| < threshold → ALIGNED
    │     else → NOT ALIGNED + direction (Move Left/Right/Up/Down)
    │
    └──── [visualization.py]  Annotate frame ───────────────────
          • Bounding boxes (hook=amber, lug=steel-blue)
          • Hook anchor   (red dot)
          • Lug centre    (cyan dot)
          • Connecting line (green=aligned, red=not)
          • dx/dy label at midpoint
          • Status badge (Aligned / Move Left / …)
```

---

## Setup

### 1. Install dependencies

```bash
cd hook_lug_app
pip install -r requirements.txt
```

### 2. Place your model

Copy your trained weights file into the project root:

```bash
cp /path/to/your/best.pt hook_lug_app/best.pt
```

### 3. Run the server

```bash
python app.py
# or with options:
python app.py --model best.pt --host 0.0.0.0 --port 5000
```

Open your browser at **http://localhost:5000**

---

## REST API

| Method | Endpoint                    | Description                                   |
|--------|-----------------------------|-----------------------------------------------|
| GET    | `/`                         | Serve web UI                                  |
| GET    | `/health`                   | Server status + GPU availability              |
| POST   | `/process_image`            | Process a single image, return annotated JPEG |
| POST   | `/upload_video`             | Upload video, start background processing     |
| GET    | `/video_feed/<vid_id>`      | MJPEG stream of annotated frames              |
| GET    | `/video_status/<vid_id>`    | Poll alignment JSON for latest frame          |
| POST   | `/stop_video/<vid_id>`      | Stop + cleanup a video stream                 |

### `POST /process_image`

**Form fields:**

| Field       | Type  | Default | Description                     |
|-------------|-------|---------|---------------------------------|
| `file`      | File  | —       | Image file (JPG/PNG/BMP/WEBP)   |
| `threshold` | int   | 30      | Alignment pixel threshold       |
| `conf`      | float | 0.25    | YOLO confidence threshold       |

**Response JSON:**
```json
{
  "image":      "<base64-encoded JPEG>",
  "alignments": [
    {
      "hook_conf": 0.92,
      "lug_conf":  0.88,
      "dx":        -12.5,
      "dy":         4.0,
      "distance":  13.1,
      "status":    "ALIGNED",
      "direction": "Aligned"
    }
  ],
  "num_pairs": 1,
  "overall":   "ALIGNED"
}
```

### `POST /upload_video`

**Form fields:** same as image plus `smooth_n` (int, default 5).

**Response JSON:**
```json
{ "video_id": "a3f9c2b1e4d7" }
```

Then open `GET /video_feed/<video_id>` in an `<img>` tag for live MJPEG stream,
and poll `GET /video_status/<video_id>` for alignment data.

---

## Configuration

| Parameter   | UI Control           | Default | Range    | Effect                             |
|-------------|----------------------|---------|----------|------------------------------------|
| Threshold   | Slider               | 30 px   | 5–150    | Pixel radius to call "aligned"     |
| Confidence  | Slider               | 0.25    | 0.05–0.95| YOLO minimum detection confidence  |
| Smoothing   | Slider (video only)  | 5       | 1–15     | (DEPRECATED) Frames to average for tip position |

---

## Model Classes

Your `best.pt` must detect exactly two classes:

| Class ID | Label  |
|----------|--------|
| 0        | hook   |
| 1        | lug    |

---

## Visual Legend

| Element          | Colour     | Meaning                        |
|------------------|------------|--------------------------------|
| Bounding box     | Amber      | Hook detection                 |
| Bounding box     | Steel-blue | Lug detection                  |
| Filled dot       | Red        | Hook tip (contour-detected)    |
| Filled dot       | Cyan       | Lug centre (bbox centre)       |
| Connecting line  | Green      | Hook–Lug pair is aligned       |
| Connecting line  | Red        | Hook–Lug pair is NOT aligned   |
| Status badge     | Green bg   | "Aligned"                      |
| Status badge     | Red bg     | "Move Left/Right/Up/Down"      |

---

## Performance Tips

- **GPU**: Automatically used when CUDA is available. Check `/health` to confirm.
- **Confidence threshold**: Raise to 0.4–0.6 to filter false positives.
- **Video frame rate**: Capped at ~30 fps via a 30 ms sleep in the MJPEG generator; reduce for slower hardware.
- **Resolution**: Resize large frames in `app.py → process_frame()` before inference if speed is critical.

---

## Extending

**Per-hook tracking across frames**: replace the per-frame processing in `VideoProcessor._run()` with a SORT/ByteTrack tracker to maintain consistent IDs and smoother anchor traces.

**Multiple model support**: pass `model_path` as a form field to `/process_image` and use `HookLugDetector(model_path)` per request (with caching by path).
