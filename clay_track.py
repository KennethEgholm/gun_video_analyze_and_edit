"""Per-clay-type clay tracker and hit/miss classifier.

Pipeline:
  1. For each frame, find the red sight bead (saturated red dot, ~3-6px).
     The bead is moved by the shooter as they swing the gun.
  2. Define a search zone relative to the bead, parameterised by clay
     type (away / coming / left / right).
  3. In each pre-shot frame, find a clay-sized dark blob in the search
     zone using a band-pass darkness filter (small-scale dark minus
     large-scale dark, which rejects hills).
  4. Greedy backward link to build a track, predict post-shot positions,
     then measure post-shot persistence and fragment count.
  5. Per-type decision rule -> hit/miss.

Public API:
  classify_shot(video_path, shot_ts, clay_type) -> dict
"""
import json
import os
import subprocess

import numpy as np

ANALYSIS_W = 640


# --- low-level helpers ------------------------------------------------------

def get_video_info(path):
    info = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True,
    )
    d = json.loads(info.stdout)
    s = d["streams"][0]
    fps = eval(s["r_frame_rate"])
    return int(s["width"]), int(s["height"]), fps


def load_window_rgb(path, t_start, t_end, width=ANALYSIS_W):
    ow, oh, fps = get_video_info(path)
    h = int(width * oh / ow); h += h % 2
    t_start = max(0.0, t_start)
    duration = max(0.0, t_end - t_start)
    cmd = ["ffmpeg", "-ss", f"{t_start:.3f}", "-i", path,
           "-t", f"{duration:.3f}",
           "-vf", f"scale={width}:-2,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    raw = proc.stdout.read(); proc.wait()
    fs = width * h * 3
    n = len(raw) // fs
    return (np.frombuffer(raw[:n * fs], dtype=np.uint8)
            .reshape(n, h, width, 3), h, fps)


def box_sum(arr, k):
    a = arr.astype(np.int32)
    h, w = a.shape
    cs = np.cumsum(np.cumsum(a, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    half = k // 2
    yy, xx = np.indices((h, w))
    y0 = np.clip(yy - half, 0, h)
    y1 = np.clip(yy + half + 1, 0, h)
    x0 = np.clip(xx - half, 0, w)
    x1 = np.clip(xx + half + 1, 0, w)
    return (cs[y1, x1] - cs[y0, x1] - cs[y1, x0] + cs[y0, x0],
            (y1 - y0) * (x1 - x0))


# --- bead detection ---------------------------------------------------------

def find_bead(rgb, search_pad=80):
    """Locate the red sight bead in an RGB frame.

    The bead is a small saturated red dot; we score pixels by
    (R - max(G, B)) and return the centroid of the brightest region.
    Returns (cy, cx) or None.
    """
    h, w = rgb.shape[:2]
    # Restrict to a roughly central region (the bead is always close to
    # center of frame).
    y0 = h // 2 - 100; y1 = h // 2 + 100
    x0 = w // 2 - search_pad; x1 = w // 2 + search_pad
    sub = rgb[y0:y1, x0:x1].astype(np.int16)
    redness = sub[..., 0] - np.maximum(sub[..., 1], sub[..., 2])
    redness = np.maximum(redness, 0)
    if redness.max() < 60:
        return None
    # Smooth slightly to find the centroid.
    s, a = box_sum(redness, 3)
    smooth = s / a
    cy, cx = np.unravel_index(int(smooth.argmax()), smooth.shape)
    return int(cy + y0), int(cx + x0)


# --- clay-score map (band-pass) --------------------------------------------

def clay_score_map(gray):
    """Highlight small (clay-sized) dark blobs while suppressing large
    (hill, shadow) dark regions."""
    sums, areas = box_sum(gray, 51)
    bg_mean = sums / areas
    dark = np.maximum(0, bg_mean - gray.astype(np.float32))
    sd, ad = box_sum(dark, 5)
    small = sd / ad
    ld, la = box_sum(dark, 41)
    large = ld / la
    return np.maximum(0, small - large)


# --- per-type search zones --------------------------------------------------

# Each entry returns a (y0, y1, x0, x1) clipped bbox relative to the bead.
ZONES = {
    # Going-away: clay flies away into hills above & around bead.
    # Wide vertical search, narrow lateral.
    "away":   {"dy_lo": -150, "dy_hi":  60, "dx_lo":  -80, "dx_hi":   80},
    # Coming: clay descends from above toward camera, often near bead.
    "coming": {"dy_lo":  -80, "dy_hi":  80, "dx_lo": -100, "dx_hi":  100},
    # Crosser moving right-to-left (bead leads to the left, clay trails right).
    "left":   {"dy_lo":  -50, "dy_hi":  60, "dx_lo":   30, "dx_hi":  300},
    # Crosser moving left-to-right (bead leads to the right, clay trails left).
    "right":  {"dy_lo":  -50, "dy_hi":  60, "dx_lo": -300, "dx_hi":  -30},
}

BEAD_MASK_RADIUS = 25  # exclude this many px around the bead


def search_mask(h, w, bead_yx, clay_type):
    yy, xx = np.indices((h, w))
    z = ZONES[clay_type]
    by, bx = bead_yx
    in_zone = ((yy - by >= z["dy_lo"]) & (yy - by <= z["dy_hi"]) &
               (xx - bx >= z["dx_lo"]) & (xx - bx <= z["dx_hi"]))
    not_bead = ((yy - by) ** 2 + (xx - bx) ** 2) > BEAD_MASK_RADIUS ** 2
    in_frame = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    return in_zone & not_bead & in_frame


def find_clay_in_zone(gray, bead_yx, clay_type, min_score=3.0):
    h, w = gray.shape
    smap = clay_score_map(gray)
    mask = search_mask(h, w, bead_yx, clay_type)
    smap = np.where(mask, smap, 0)
    score = float(smap.max())
    if score < min_score:
        return None
    cy, cx = np.unravel_index(int(smap.argmax()), smap.shape)
    return int(cy), int(cx), score


# --- tracking ---------------------------------------------------------------

def track_clay(rgb_frames, gray_frames, clay_type,
               n_pre=12, max_jump=60):
    """Build a backwards-link track of the clay across pre-shot frames.

    Returns (track, beads) where:
      track = [(frame_idx, cy, cx, score), ...]
      beads = [(cy, cx) | None per frame]
    """
    n = len(gray_frames)
    beads = [find_bead(rgb_frames[i]) for i in range(n)]
    detections = []
    for i in range(n):
        if beads[i] is None:
            detections.append(None)
            continue
        d = find_clay_in_zone(gray_frames[i], beads[i], clay_type)
        detections.append(d)
    seed = next((i for i in range(n - 1, -1, -1)
                 if detections[i] is not None), None)
    if seed is None:
        return [], beads
    cy, cx, sc = detections[seed]
    track = [(seed, cy, cx, sc)]
    last = seed
    start = max(0, seed - n_pre)
    for i in range(seed - 1, start - 1, -1):
        c = detections[i]
        if c is None:
            if last - i > 2:
                break
            continue
        d = ((c[0] - cy) ** 2 + (c[1] - cx) ** 2) ** 0.5
        if d > max_jump:
            if last - i > 2:
                break
            continue
        track.insert(0, (i, c[0], c[1], c[2]))
        cy, cx = c[0], c[1]
        last = i
    return track, beads


def predict(track, target_idx):
    if len(track) < 2:
        return None
    use = track[-3:] if len(track) >= 3 else track[-2:]
    idxs = np.array([p[0] for p in use], dtype=float)
    cys = np.array([p[1] for p in use], dtype=float)
    cxs = np.array([p[2] for p in use], dtype=float)
    A = np.vstack([idxs, np.ones_like(idxs)]).T
    my, by = np.linalg.lstsq(A, cys, rcond=None)[0]
    mx, bx = np.linalg.lstsq(A, cxs, rcond=None)[0]
    return float(my * target_idx + by), float(mx * target_idx + bx)


def find_clay_near(gray, cy, cx, bead_yx, half=25, min_score=2.0):
    """Find the strongest clay-score blob within a small box around (cy, cx),
    excluding the bead."""
    h, w = gray.shape
    sy0 = max(0, cy - half); sy1 = min(h, cy + half + 1)
    sx0 = max(0, cx - half); sx1 = min(w, cx + half + 1)
    sub = gray[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return None
    smap = clay_score_map(sub)
    if bead_yx is not None:
        by, bx = bead_yx
        yy, xx = np.indices(gray.shape)
        bead_glob = ((yy - by) ** 2 + (xx - bx) ** 2) <= BEAD_MASK_RADIUS ** 2
        smap[bead_glob[sy0:sy1, sx0:sx1]] = 0
    score = float(smap.max())
    if score < min_score:
        return None, score
    ly, lx = np.unravel_index(int(smap.argmax()), smap.shape)
    return (int(ly + sy0), int(lx + sx0)), score


# --- post-shot evaluation --------------------------------------------------

def post_shot_metrics(gray_frames, beads, track, shot_idx,
                     n_post=8, search_half=30):
    """Continue the track forward through n_post frames; record per-frame
    score, distance from prediction, and an n_blobs count of distinct
    clay-score peaks in a wider area."""
    metrics = []
    for off in range(1, n_post + 1):
        tgt = shot_idx + off
        if tgt >= len(gray_frames):
            break
        pred = predict(track, tgt)
        if pred is None:
            break
        py, px = pred
        bead = beads[tgt] if tgt < len(beads) else None
        # Find blob near prediction.
        near = find_clay_near(gray_frames[tgt], int(py), int(px), bead,
                              half=search_half)
        if near is None or near[0] is None:
            score = near[1] if near is not None else 0.0
            metrics.append({"off": off, "found": False, "score": score,
                            "dist": 999.0, "n_blobs": 0})
            continue
        (cy, cx), score = near
        dist = ((cy - py) ** 2 + (cx - px) ** 2) ** 0.5
        # Count blobs in a wider area around the prediction.
        n_blobs = count_blobs(gray_frames[tgt], int(py), int(px), bead,
                              half=60)
        metrics.append({"off": off, "found": True, "score": score,
                        "dist": dist, "n_blobs": n_blobs,
                        "cy": cy, "cx": cx})
    return metrics


def count_blobs(gray, cy, cx, bead_yx, half=60, min_score=3.0):
    h, w = gray.shape
    sy0 = max(0, cy - half); sy1 = min(h, cy + half + 1)
    sx0 = max(0, cx - half); sx1 = min(w, cx + half + 1)
    sub = gray[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return 0
    smap = clay_score_map(sub)
    if bead_yx is not None:
        by, bx = bead_yx
        yy, xx = np.indices(gray.shape)
        bead_glob = ((yy - by) ** 2 + (xx - bx) ** 2) <= BEAD_MASK_RADIUS ** 2
        smap[bead_glob[sy0:sy1, sx0:sx1]] = 0
    work = smap.copy()
    n = 0
    for _ in range(15):
        if float(work.max()) < min_score:
            break
        n += 1
        py, px = np.unravel_index(int(work.argmax()), work.shape)
        y0 = max(0, py - 5); y1 = min(work.shape[0], py + 6)
        x0 = max(0, px - 5); x1 = min(work.shape[1], px + 6)
        work[y0:y1, x0:x1] = 0
    return n


# --- main entry point -------------------------------------------------------

def classify_shot(video_path, shot_ts, clay_type,
                  pre_window=0.30, post_window=0.40):
    """Run the full pipeline for one shot. Returns a metrics dict."""
    t_start = max(0.0, shot_ts - pre_window)
    t_end = shot_ts + post_window
    rgb, h, fps = load_window_rgb(video_path, t_start, t_end)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2]).astype(np.uint8)
    if len(gray) == 0:
        return {"ok": False, "reason": "no_frames"}
    shot_idx = int(round((shot_ts - t_start) * fps))
    shot_idx = min(shot_idx, len(gray) - 1)

    # Track only on pre-shot frames.
    track, beads = track_clay(rgb[:shot_idx + 1], gray[:shot_idx + 1],
                              clay_type)
    # Also detect beads for post-shot frames so post-shot search can mask them.
    for i in range(shot_idx + 1, len(gray)):
        beads.append(find_bead(rgb[i]))

    if len(track) < 2:
        return {"ok": False, "reason": "no_track", "n_track": len(track),
                "beads_pre": [b for b in beads[:shot_idx + 1] if b]}

    post = post_shot_metrics(gray, beads, track, shot_idx)
    pre_score = float(np.median([p[3] for p in track[-3:]]))

    return {
        "ok": True, "n_track": len(track),
        "track_last": track[-1][1:],
        "pre_score": pre_score,
        "post": post,
        "shot_idx": shot_idx, "n_frames": len(gray),
        "fps": fps,
    }


# --- per-type decision rules -----------------------------------------------

# Threshold trained on videos/training: misses for crossers have
# min_score_late / pre_score < 0.25 (linear extrapolation fails because
# the intact clay arcs downward away from the prediction); hits keep
# ratio >= 0.32 (broken fragments remain near the break point).
CROSSER_RATIO_HIT = 0.25


def _ratio_metric(metrics):
    """Return min_late_score / pre_score, or 1.0 if unavailable.

    "Late" = post-shot offsets >= 5 frames (≈125ms at 40fps)."""
    post = metrics.get("post") or []
    sc_late = [p["score"] for p in post if p["off"] >= 5 and p["found"]]
    if not sc_late or metrics.get("pre_score", 0) <= 0:
        # No persistent dark blob found near prediction -> miss (clay
        # flew on but linear extrapolation lost it).
        return 0.0
    return min(sc_late) / max(1.0, metrics["pre_score"])


def classify_outcome(metrics, clay_type):
    """Decide hit vs miss from track metrics + clay_type.

    Returns dict {is_hit: bool, reason: str, score: float}.

    Note: the "away" path here is a fallback only -- the existing
    block-based detector in detect_hits.py handles going-away clays
    much better. Callers should use the block detector for "away".
    """
    if not metrics.get("ok"):
        # Lost the clay before the shot -> can't decide; default to miss
        # (no evidence of fragmentation visible).
        return {"is_hit": False, "reason": metrics.get("reason", "no_metrics"),
                "score": 0.0}

    if clay_type in ("left", "right"):
        ratio = _ratio_metric(metrics)
        return {"is_hit": ratio >= CROSSER_RATIO_HIT,
                "reason": f"crosser_ratio={ratio:.2f}",
                "score": ratio}

    if clay_type == "coming":
        # Limited training data (no misses). Apply same rule.
        ratio = _ratio_metric(metrics)
        return {"is_hit": ratio >= CROSSER_RATIO_HIT,
                "reason": f"coming_ratio={ratio:.2f}",
                "score": ratio}

    # away: tracker can't separate; default to hit so callers know to
    # fall back to the block-based detector.
    return {"is_hit": True, "reason": "away_use_block_detector",
            "score": 1.0}

