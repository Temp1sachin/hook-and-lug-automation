/* ─────────────────────────────────────────────────────────────
  S.I.G.H.T.  •  main.js
   ───────────────────────────────────────────────────────────── */

"use strict";

/* ── State ───────────────────────────────────────────────────── */
const STATE = {
  mode:        "image",   // "image" | "video"
  videoId:     null,
  pollTimer:   null,
  resultB64:   null,      // base64 of last image result
};

/* ── DOM refs ─────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const els = {
  fileInput:      $("file-input"),
  uploadZone:     $("upload-zone"),
  dropTarget:     $("drop-target"),
  uploadHint:     $("upload-hint"),
  resultZone:     $("result-zone"),
  resultTitle:    $("result-title"),
  resultImg:      $("result-img"),
  videoStream:    $("video-stream"),
  resultCanvas:   $("result-canvas"),
  canvasOverlay:  $("canvas-overlay"),
  loadingOverlay: $("loading-overlay"),
  loadingText:    $("loading-text"),
  statusList:     $("status-list"),
  statsPanel:     $("stats-panel"),
  statPairs:      $("stat-pairs"),
  statOverall:    $("stat-overall"),
  statDist:       $("stat-dist"),
  statThr:        $("stat-thr"),
  liveBadge:      $("live-badge"),
  videoBar:       $("video-bar"),
  videoInfo:      $("video-info"),
  downloadBtn:    $("download-btn"),
  smoothRow:      $("smooth-row"),
  deviceText:     $("device-text"),
  deviceBadge:    $("device-badge"),
  toast:          $("toast"),
};

/* ── Boot ─────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  setupDragDrop();
  setupFileInput();
  checkHealth();
});

/* ── Health check ─────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    if (d.gpu) {
      els.deviceText.textContent = "GPU";
      els.deviceBadge.style.color = "var(--amber)";
    }
  } catch (_) {}
}

/* ── Mode switching ───────────────────────────────────────────── */
function setMode(mode) {
  STATE.mode = mode;
  $("btn-image").classList.toggle("active", mode === "image");
  $("btn-video").classList.toggle("active", mode === "video");
  els.smoothRow.style.display = mode === "video" ? "block" : "none";
  els.uploadHint.textContent  = mode === "image"
    ? "JPG · PNG · BMP · WEBP"
    : "MP4 · AVI · MOV · MKV";
  els.fileInput.accept = mode === "image"
    ? ".jpg,.jpeg,.png,.bmp,.webp"
    : ".mp4,.avi,.mov,.mkv";
  resetUI();
}

/* ── Slider display helpers ───────────────────────────────────── */
function updateVal(displayId, value, suffix) {
  $(displayId).textContent = value + suffix;
  if (displayId === "thr-display") els.statThr.textContent = value + " px";
}

/* ── Drag-and-drop ────────────────────────────────────────────── */
function setupDragDrop() {
  const dt = els.dropTarget;

  ["dragenter","dragover"].forEach(evt =>
    dt.addEventListener(evt, e => { e.preventDefault(); dt.classList.add("drag-over"); })
  );

  ["dragleave","dragend","drop"].forEach(evt =>
    dt.addEventListener(evt, e => { e.preventDefault(); dt.classList.remove("drag-over"); })
  );

  dt.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  dt.addEventListener("click", () => els.fileInput.click());
}

/* ── File input ───────────────────────────────────────────────── */
function setupFileInput() {
  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files[0];
    if (file) handleFile(file);
    els.fileInput.value = "";
  });
}

/* ── Route file to appropriate handler ───────────────────────── */
function handleFile(file) {
  const isVideo = /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
  const isImage = /\.(jpe?g|png|bmp|webp)$/i.test(file.name);

  if (STATE.mode === "image" && !isImage) {
    return showToast("Switch to Video mode for video files", "error");
  }
  if (STATE.mode === "video" && !isVideo) {
    return showToast("Switch to Image mode for image files", "error");
  }

  if (STATE.mode === "image") uploadImage(file);
  else                        uploadVideo(file);
}

/* ══════════════════════════════════════════════════════════════
   IMAGE MODE
═══════════════════════════════════════════════════════════════ */
async function uploadImage(file) {
  showLoading("Running YOLO + edge detection…");

  const fd = new FormData();
  fd.append("file",      file);
  fd.append("threshold", $("threshold").value);
  fd.append("conf",      $("conf").value);

  try {
    const res  = await fetch("/process_image", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    hideLoading();
    STATE.resultB64 = data.image;
    showImageResult(data);
  } catch (err) {
    hideLoading();
    showToast("Error: " + err.message, "error");
  }
}

function showImageResult(data) {
  // Show result zone
  showResultZone("image");

  // Render annotated image
  els.resultImg.src = "data:image/jpeg;base64," + data.image;
  els.resultImg.style.display = "block";
  els.videoStream.style.display = "none";

  // Update status list
  renderStatusList(data.alignments);
  renderStats(data.alignments, data.overall);
}

/* ══════════════════════════════════════════════════════════════
   VIDEO MODE
═══════════════════════════════════════════════════════════════ */
async function uploadVideo(file) {
  showLoading("Uploading & starting stream…");

  const fd = new FormData();
  fd.append("file",      file);
  fd.append("threshold", $("threshold").value);
  fd.append("conf",      $("conf").value);
  fd.append("smooth_n",  $("smooth_n").value);

  try {
    const res  = await fetch("/upload_video", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    STATE.videoId = data.video_id;
    hideLoading();
    startVideoStream(data.video_id);
  } catch (err) {
    hideLoading();
    showToast("Error: " + err.message, "error");
  }
}

function startVideoStream(vid_id) {
  showResultZone("video");

  // Set MJPEG stream source
  els.videoStream.src = `/video_feed/${vid_id}`;
  els.videoStream.style.display = "block";
  els.resultImg.style.display   = "none";
  els.canvasOverlay.style.display = "none";

  els.liveBadge.style.display = "flex";
  els.videoBar.style.display  = "flex";
  els.videoInfo.textContent   = `Stream ID: ${vid_id}`;

  // Poll alignment data every 500 ms
  STATE.pollTimer = setInterval(() => pollVideoStatus(vid_id), 500);

  // Detect stream end
  els.videoStream.addEventListener("error", onStreamEnd, { once: true });
}

async function pollVideoStatus(vid_id) {
  try {
    const res  = await fetch(`/video_status/${vid_id}`);
    const data = await res.json();

    if (data.alignments && data.alignments.length > 0) {
      renderStatusList(data.alignments);
      renderStats(data.alignments,
        data.alignments.every(a => a.status === "ALIGNED") ? "ALIGNED" : "NOT ALIGNED"
      );
    }

    if (data.done && !data.running) onStreamEnd();
  } catch (_) {}
}

function onStreamEnd() {
  clearInterval(STATE.pollTimer);
  STATE.pollTimer = null;
  els.liveBadge.style.display = "none";
  els.videoInfo.textContent   = "Stream finished";
  els.canvasOverlay.style.display = "flex";
  showToast("Video processing complete", "success");
}

async function stopVideo() {
  if (!STATE.videoId) return;
  clearInterval(STATE.pollTimer);
  await fetch(`/stop_video/${STATE.videoId}`, { method: "POST" }).catch(() => {});
  els.videoStream.src = "";
  STATE.videoId = null;
  onStreamEnd();
}

/* ══════════════════════════════════════════════════════════════
   STATUS / STATS RENDERING
═══════════════════════════════════════════════════════════════ */
function renderStatusList(alignments) {
  if (!alignments || alignments.length === 0) {
    els.statusList.innerHTML = `<div class="empty-state">No hook-lug pairs detected</div>`;
    return;
  }

  els.statusList.innerHTML = alignments.map((al, i) => {
    const cls       = al.status === "ALIGNED"     ? "aligned"
                    : al.status === "NOT ALIGNED"  ? "not-aligned"
                    : "unknown";
    const badgeCls  = cls;
    const dirCls    = cls;
    const badgeTxt  = al.status || "UNKNOWN";
    const direction = al.direction || "—";
    const dx        = al.dx   != null ? (al.dx >= 0 ? "+" : "") + al.dx   + " mm" : "—";
    const dy        = al.dy   != null ? (al.dy >= 0 ? "+" : "") + al.dy   + " mm" : "—";
    const dist      = al.distance != null ? al.distance + " mm" : "—";

    return `
      <div class="status-card ${cls}">
        <div class="sc-header">
          <span class="sc-pair">Pair ${i + 1}</span>
          <span class="sc-badge ${badgeCls}">${badgeTxt}</span>
        </div>
        <div class="sc-direction ${dirCls}">${directionIcon(al)} ${direction}</div>
        <div class="sc-metrics">
          <div class="sc-metric">
            <div class="sc-metric-label">dx</div>
            <div class="sc-metric-val">${dx}</div>
          </div>
          <div class="sc-metric">
            <div class="sc-metric-label">dy</div>
            <div class="sc-metric-val">${dy}</div>
          </div>
          <div class="sc-metric">
            <div class="sc-metric-label">dist</div>
            <div class="sc-metric-val">${dist}</div>
          </div>
        </div>
      </div>`;
  }).join("");
}

function directionIcon(al) {
  if (al.status === "ALIGNED") return "✓";
  const dir = al.direction || "";
  if (dir.includes("Left")  && dir.includes("Up"))   return "↖";
  if (dir.includes("Right") && dir.includes("Up"))   return "↗";
  if (dir.includes("Left")  && dir.includes("Down")) return "↙";
  if (dir.includes("Right") && dir.includes("Down")) return "↘";
  if (dir.includes("Left"))  return "←";
  if (dir.includes("Right")) return "→";
  if (dir.includes("Up"))    return "↑";
  if (dir.includes("Down"))  return "↓";
  return "⚠";
}

function renderStats(alignments, overall) {
  els.statsPanel.style.display = "block";
  els.statPairs.textContent    = alignments.length;

  const overallEl = els.statOverall;
  overallEl.textContent = overall === "ALIGNED" ? "OK" :
                          overall === "NOT ALIGNED" ? "FAIL" : "N/A";
  overallEl.style.color = overall === "ALIGNED" ? "var(--green)" :
                          overall === "NOT ALIGNED" ? "var(--red)"  : "var(--grey)";

  const dists = alignments
    .filter(a => a.distance != null)
    .map(a => a.distance);
  els.statDist.textContent = dists.length
    ? Math.min(...dists).toFixed(1) + " px"
    : "—";
}

/* ══════════════════════════════════════════════════════════════
   UI HELPERS
═══════════════════════════════════════════════════════════════ */
function showResultZone(type) {
  els.uploadZone.style.display = "none";
  els.resultZone.style.display = "flex";
  els.resultTitle.textContent  = type === "video" ? "Live Detection Stream" : "Detection Output";
  els.downloadBtn.style.display = type === "image" ? "flex" : "none";
}

function showLoading(msg = "Processing…") {
  els.loadingText.textContent     = msg;
  els.loadingOverlay.style.display = "flex";
}

function hideLoading() {
  els.loadingOverlay.style.display = "none";
}

function resetUI() {
  // Stop any active video
  if (STATE.videoId) stopVideo();
  clearInterval(STATE.pollTimer);
  STATE.pollTimer = null;
  STATE.videoId   = null;
  STATE.resultB64 = null;

  // Reset video stream
  els.videoStream.src = "";

  // Show upload, hide result
  els.uploadZone.style.display  = "flex";
  els.resultZone.style.display  = "none";
  els.liveBadge.style.display   = "none";
  els.videoBar.style.display    = "none";
  els.canvasOverlay.style.display = "none";
  els.statsPanel.style.display  = "none";
  els.statusList.innerHTML      = `<div class="empty-state">Upload a file to begin detection</div>`;
  hideLoading();
}

function downloadResult() {
  if (!STATE.resultB64) return;
  const a    = document.createElement("a");
  a.href     = "data:image/jpeg;base64," + STATE.resultB64;
  a.download = "alignment_result.jpg";
  a.click();
}

/* ── Toast ────────────────────────────────────────────────────── */
let _toastTimer;
function showToast(msg, type = "") {
  const t = els.toast;
  t.textContent = msg;
  t.className   = "toast show" + (type ? " " + type : "");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
}
