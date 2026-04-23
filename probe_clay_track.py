"""Probe v3: detect the clay as a small DARK spot against bright sky.

Pipeline per shot:
  1. Load 640px-wide frames in [shot-0.30s, shot+0.40s] as grayscale.
  2. For each frame, build a "darkness" map = local_background - pixel.
     Local background is a wide box mean (~50px). A clay shows up as a
     small region with darkness > some threshold.
  3. Track the brightest darkness spot (small enough to be a clay) backwards
     from the latest pre-shot frame.
  4. Predict the position 50-150ms after the shot and check whether a
     coherent dark spot still exists (miss) or has fragmented/vanished
     (hit).

Outputs a per-shot table and saves debug overlays to /tmp/clay_probe.
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

VIDEO_DIR = "videos"
ANALYSIS_W = 640
DEBUG_DIR = "/tmp/clay_probe"
BEAD_CY, BEAD_CX = 180, 320       # gun's sight bead, fixed
BEAD_MASK_RADIUS = 35             # exclude this region from clay search
# Trail zone for left-to-right crossers: clay trails LEFT of the bead
# and may sit slightly low. Coordinates are in 640px-wide frames.
TRAIL_X_LO, TRAIL_X_HI = 30, BEAD_CX - BEAD_MASK_RADIUS
TRAIL_Y_LO, TRAIL_Y_HI = BEAD_CY - 80, BEAD_CY + 50
TRUTH = json.load(open(os.path.join(VIDEO_DIR, "hits_feedback.json")))
TRUTH_SHOTS = {(e["file"], round(e["timestamp"], 2)): e["outcome"]
               for e in TRUTH if e["type"] == "shot"}


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


def load_window(path, t_start, t_end, width=ANALYSIS_W):
    """Load grayscale frames in [t_start, t_end]. Returns (frames, h, fps)."""
    ow, oh, fps = get_video_info(path)
    h = int(width * oh / ow); h += h % 2
    t_start = max(0.0, t_start)
    duration = max(0.0, t_end - t_start)
    cmd = ["ffmpeg", "-ss", f"{t_start:.3f}", "-i", path,
           "-t", f"{duration:.3f}",
           "-vf", f"scale={width}:-2,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-v", "quiet", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    raw = proc.stdout.read(); proc.wait()
    fs = width * h
    n = len(raw) // fs
    frames = np.frombuffer(raw[:n * fs], dtype=np.uint8) \
        .reshape(n, h, width)
    return frames, h, fps


def load_window_rgb(path, t_start, t_end, width=ANALYSIS_W):
    """For debug overlays only."""
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
    return np.frombuffer(raw[:n * fs], dtype=np.uint8).reshape(n, h, width, 3)


def box_sum(arr, k):
    """Box-sum via integral image."""
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
    return cs[y1, x1] - cs[y0, x1] - cs[y1, x0] + cs[y0, x0], \
        (y1 - y0) * (x1 - x0)


def darkness_map(gray, bg_k=51, blob_k=5, static_bg=None):
    """Positive where pixel is darker than the wide-area local mean."""
    sums, areas = box_sum(gray, bg_k)
    bg_mean = sums / areas
    dark = bg_mean - gray.astype(np.float32)
    dark = np.maximum(0, dark)
    if static_bg is not None:
        sums_bg, _ = box_sum(static_bg, bg_k)
        bg_mean_bg = sums_bg / areas
        dark_bg = np.maximum(0, bg_mean_bg - static_bg.astype(np.float32))
        dark = np.maximum(0, dark - dark_bg)
    sd, ad = box_sum(dark, blob_k)
    return sd / ad


def build_bead_mask(h, w):
    """Boolean mask: True OUTSIDE the bead exclusion zone."""
    yy, xx = np.indices((h, w))
    return ((yy - BEAD_CY) ** 2 + (xx - BEAD_CX) ** 2) > BEAD_MASK_RADIUS ** 2


def change_map(prev_gray, curr_gray, blob_k=5):
    """Smoothed |curr - prev| highlighting moving features."""
    diff = np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16))
    sd, ad = box_sum(diff, blob_k)
    return (sd / ad).astype(np.float32)


def trail_mask(h, w):
    """Boolean mask: True INSIDE the left-trail crosser search zone
    (and outside the bead)."""
    yy, xx = np.indices((h, w))
    in_zone = ((xx >= TRAIL_X_LO) & (xx <= TRAIL_X_HI) &
               (yy >= TRAIL_Y_LO) & (yy <= TRAIL_Y_HI))
    not_bead = ((yy - BEAD_CY) ** 2 + (xx - BEAD_CX) ** 2) > BEAD_MASK_RADIUS ** 2
    return in_zone & not_bead


def clay_score_map(gray):
    """Map favouring small (clay-sized) dark blobs over large (hill)
    dark regions. Computed as a band-pass on the darkness map.
    """
    sums, areas = box_sum(gray, 51)
    bg_mean = sums / areas
    dark = np.maximum(0, bg_mean - gray.astype(np.float32))
    # Small-scale darkness (clay-sized).
    sd, ad = box_sum(dark, 5)
    small = sd / ad
    # Large-scale darkness (hills, big shadows).
    ld, la = box_sum(dark, 41)
    large = ld / la
    # Band-pass: positive where small-scale dark exceeds large-scale.
    return np.maximum(0, small - large)


def find_dark_in_trail(gray, min_dark=3.0):
    """Return (cy, cx, score) of the strongest small-scale dark blob in
    the trail zone (band-pass filtered to reject hills).
    """
    h, w = gray.shape
    smap = clay_score_map(gray)
    mask = trail_mask(h, w)
    smap = np.where(mask, smap, 0)
    score = float(smap.max())
    if score < min_dark:
        return None
    cy, cx = np.unravel_index(int(smap.argmax()), smap.shape)
    return int(cy), int(cx), score


def find_clay_track(frames):
    """Track clay backwards. For crossers we look for a dark blob in the
    trail zone in each pre-shot frame and link by proximity."""
    n = len(frames)
    detections = [find_dark_in_trail(f) for f in frames]
    seed_idx = next((i for i in range(n - 1, -1, -1)
                     if detections[i] is not None), None)
    if seed_idx is None:
        return []
    cy, cx, _ = detections[seed_idx]
    track = [(seed_idx, cy, cx)]
    last = seed_idx
    for i in range(seed_idx - 1, -1, -1):
        c = detections[i]
        if c is None:
            if last - i > 2:
                break
            continue
        d = ((c[0] - cy) ** 2 + (c[1] - cx) ** 2) ** 0.5
        if d > 60:
            if last - i > 2:
                break
            continue
        track.insert(0, (i, c[0], c[1]))
        cy, cx = c[0], c[1]
        last = i
    return track


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


def track_post_shot(frames, pre_track, n_post=8, search_half=25,
                    min_dark=2.0):
    """Continue the pre-shot track forward through post-shot frames.

    Returns a list of (offset, found, cy, cx, score, dist_from_pred) for
    each frame from +1..+n_post. found=False means we lost the clay there.
    """
    if len(pre_track) < 2:
        return []
    use = pre_track[-3:] if len(pre_track) >= 3 else pre_track[-2:]
    idxs = np.array([p[0] for p in use], dtype=float)
    cys = np.array([p[1] for p in use], dtype=float)
    cxs = np.array([p[2] for p in use], dtype=float)
    A = np.vstack([idxs, np.ones_like(idxs)]).T
    my, by = np.linalg.lstsq(A, cys, rcond=None)[0]
    mx, bx = np.linalg.lstsq(A, cxs, rcond=None)[0]
    last_idx, last_cy, last_cx = pre_track[-1]
    out = []
    for off in range(1, n_post + 1):
        tgt = last_idx + off
        if tgt >= len(frames):
            break
        pred_y = my * tgt + by
        pred_x = mx * tgt + bx
        h, w = frames[tgt].shape
        # Search a tight box around prediction
        cy, cx = int(round(pred_y)), int(round(pred_x))
        sy0 = max(0, cy - search_half); sy1 = min(h, cy + search_half + 1)
        sx0 = max(0, cx - search_half); sx1 = min(w, cx + search_half + 1)
        sub = frames[tgt][sy0:sy1, sx0:sx1]
        if sub.size == 0:
            out.append((off, False, cy, cx, 0.0, 999.0))
            continue
        dmap = clay_score_map(sub)
        # Mask the bead in the sub
        yy, xx = np.indices(frames[tgt].shape)
        bead_glob = ((yy - BEAD_CY) ** 2 + (xx - BEAD_CX) ** 2) <= BEAD_MASK_RADIUS ** 2
        dmap[bead_glob[sy0:sy1, sx0:sx1]] = 0
        score = float(dmap.max())
        if score < min_dark:
            out.append((off, False, cy, cx, score, 999.0))
            continue
        ly, lx = np.unravel_index(int(dmap.argmax()), dmap.shape)
        gcy, gcx = ly + sy0, lx + sx0
        dist = ((gcy - cy) ** 2 + (gcx - cx) ** 2) ** 0.5
        out.append((off, True, gcy, gcx, score, dist))
    return out


def evaluate_post(curr_gray, predicted_cy, predicted_cx,
                  inner_half=12, search_half=40, min_dark=4.0):
    """Look for a dark blob near the predicted post-shot position.

    On a MISS: clay continues -> coherent dark blob near predicted spot.
    On a HIT:  clay shatters/puffs -> blob vanishes or fragments.
    """
    h, w = curr_gray.shape
    cy, cx = int(round(predicted_cy)), int(round(predicted_cx))
    sy0 = max(0, cy - search_half); sy1 = min(h, cy + search_half + 1)
    sx0 = max(0, cx - search_half); sx1 = min(w, cx + search_half + 1)
    sub = curr_gray[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return None
    dmap = darkness_map(sub)
    # Mask the bead region inside the sub if it overlaps
    yy, xx = np.indices(curr_gray.shape)
    bead_glob = ((yy - BEAD_CY) ** 2 + (xx - BEAD_CX) ** 2) <= BEAD_MASK_RADIUS ** 2
    dmap[bead_glob[sy0:sy1, sx0:sx1]] = 0

    icy = cy - sy0; icx = cx - sx0
    iy0 = max(0, icy - inner_half); iy1 = min(dmap.shape[0], icy + inner_half + 1)
    ix0 = max(0, icx - inner_half); ix1 = min(dmap.shape[1], icx + inner_half + 1)
    roi_max = float(dmap[iy0:iy1, ix0:ix1].max()) if (iy1 > iy0 and ix1 > ix0) else 0.0
    best = float(dmap.max())
    by_, bx_ = np.unravel_index(int(dmap.argmax()), dmap.shape)
    by_g, bx_g = by_ + sy0, bx_ + sx0
    dist = ((by_g - cy) ** 2 + (bx_g - cx) ** 2) ** 0.5

    work = dmap.copy()
    n_blobs = 0
    for _ in range(8):
        m = float(work.max())
        if m < min_dark:
            break
        n_blobs += 1
        py, px = np.unravel_index(int(work.argmax()), work.shape)
        y0 = max(0, py - 4); y1 = min(work.shape[0], py + 5)
        x0 = max(0, px - 4); x1 = min(work.shape[1], px + 5)
        work[y0:y1, x0:x1] = 0
    return {
        "roi_max": roi_max, "best": best, "best_dist": dist,
        "best_cy": int(by_g), "best_cx": int(bx_g),
        "n_blobs": n_blobs,
    }


def draw_debug(frame_rgb, track, predicted, post_eval, label, out_path,
               inner_half=12, search_half=40):
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    # Trail search zone (cyan) and bead exclusion (magenta)
    draw.rectangle([TRAIL_X_LO, TRAIL_Y_LO, TRAIL_X_HI, TRAIL_Y_HI],
                   outline=(0, 200, 200))
    draw.ellipse([BEAD_CX - BEAD_MASK_RADIUS, BEAD_CY - BEAD_MASK_RADIUS,
                  BEAD_CX + BEAD_MASK_RADIUS, BEAD_CY + BEAD_MASK_RADIUS],
                 outline=(255, 0, 255))
    for (_, cy, cx) in track:
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], outline=(0, 255, 0))
    if predicted is not None:
        py, px = predicted
        draw.line([px - 6, py - 6, px + 6, py + 6], fill=(255, 255, 0), width=2)
        draw.line([px - 6, py + 6, px + 6, py - 6], fill=(255, 255, 0), width=2)
        draw.rectangle([px - inner_half, py - inner_half,
                        px + inner_half, py + inner_half],
                       outline=(255, 255, 0))
        draw.rectangle([px - search_half, py - search_half,
                        px + search_half, py + search_half],
                       outline=(255, 100, 0))
    if post_eval is not None:
        bx_, by_ = post_eval["best_cx"], post_eval["best_cy"]
        draw.ellipse([bx_ - 6, by_ - 6, bx_ + 6, by_ + 6], outline=(255, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))
    img.save(out_path)


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    shots_by_file = {}
    for (f, t), outcome in TRUTH_SHOTS.items():
        shots_by_file.setdefault(f, []).append((t, outcome))

    print(f"{'file':<14} {'ts':>6} {'out':>5}  "
          f"{'tlen':>4}  pre(y,x)   "
          f"post: +1..+8 (./* found, score)  hits  longest_run")
    rows = []
    for vname in sorted(shots_by_file):
        path = os.path.join(VIDEO_DIR, vname)
        for ts, outcome in sorted(shots_by_file[vname]):
            t_start = ts - 0.30
            t_end = ts + 0.40
            gray, h, fps = load_window(path, t_start, t_end)
            if len(gray) == 0:
                continue
            shot_idx = int(round((ts - t_start) * fps))
            shot_idx = min(shot_idx, len(gray) - 1)

            pre = gray[:shot_idx + 1]
            track = find_clay_track(pre)
            if not track:
                print(f"{vname:<14} {ts:>6.2f} {outcome:>5}  "
                      f"{0:>4}  NO_PRE_CLAY")
                continue

            last = track[-1]
            # Pre-shot baseline: take darkness score at the last few
            # pre-shot detections.
            pre_scores = []
            for (i, cy, cx) in track[-3:]:
                d = darkness_map(gray[i])
                pre_scores.append(float(d[cy, cx]))
            pre_score = float(np.median(pre_scores)) if pre_scores else 0.0

            # Continue tracking through post-shot frames.
            post = track_post_shot(gray, track, n_post=8)
            # Compact display: ".sNN" if found, "x" if lost.
            cells = []
            for off, found, cy, cx, score, dist in post:
                if found:
                    cells.append(f".{score:>4.1f}/{dist:>3.0f}")
                else:
                    cells.append(f"x         ")
            line = " ".join(cells)
            n_found = sum(1 for _, f, *_ in post if f)
            # Longest run of consecutive found
            longest = cur = 0
            for _, f, *_ in post:
                if f:
                    cur += 1; longest = max(longest, cur)
                else:
                    cur = 0
            print(f"{vname:<14} {ts:>6.2f} {outcome:>5}  "
                  f"{len(track):>4}  ({last[1]:>3},{last[2]:>3})  "
                  f"pre={pre_score:>4.1f}  "
                  f"{line}  found={n_found}  run={longest}")
            rows.append({
                "file": vname, "ts": ts, "outcome": outcome,
                "track_len": len(track),
                "n_found": n_found, "longest_run": longest,
                "post": [{"off": o, "found": f, "score": s, "dist": d}
                         for o, f, _, _, s, d in post],
            })

            # Debug image (RGB) at +3f -- redo a tiny inline evaluate_post
            # just for display.
            if len(post) >= 3:
                rgb = load_window_rgb(path, t_start, t_end)
                tgt = shot_idx + 3
                if 0 <= tgt < len(rgb):
                    use = track[-3:] if len(track) >= 3 else track[-2:]
                    idxs = np.array([p[0] for p in use], dtype=float)
                    cys = np.array([p[1] for p in use], dtype=float)
                    cxs = np.array([p[2] for p in use], dtype=float)
                    A = np.vstack([idxs, np.ones_like(idxs)]).T
                    my, by = np.linalg.lstsq(A, cys, rcond=None)[0]
                    mx, bx = np.linalg.lstsq(A, cxs, rcond=None)[0]
                    pred = (my * tgt + by, mx * tgt + bx)
                    ev = evaluate_post(gray[tgt], pred[0], pred[1])
                    draw_debug(rgb[tgt], track, pred, ev,
                               f"{vname} ts={ts:.2f} {outcome} +3f",
                               os.path.join(DEBUG_DIR,
                                            f"{vname[:-4]}_{ts:.2f}_{outcome}.png"))

    json.dump(rows, open("/tmp/clay_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
