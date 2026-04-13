#!/usr/bin/env python3
"""
Analyze hit vs miss patterns in clay target shooting videos.

Usage:
    python3 analyze_shots.py [VIDEO_DIR]

Reads hits.json (from detect_hits.py detect --type both) and produces
a statistical comparison of hit vs miss shots with text report and plots.
"""

import argparse
import io
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Import reusable functions from detect_hits.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_hits import (
    get_video_info,
    extract_audio,
    compute_audio_energy,
    ANALYSIS_WIDTH,
    AUDIO_SAMPLE_RATE,
    AUDIO_WINDOW_MS,
    HITS_FILE,
)

# ── constants ───────────────────────────────────────────────────────────────

WINDOW_PRE = 1.0          # seconds before shot for profiles
WINDOW_POST = 1.0         # seconds after shot for profiles
STABILITY_WINDOW = 1.0    # seconds before shot for stability variance
HIT_PAIR_MAX_DELAY = 3.0  # max seconds after shot to look for a hit
HIT_PAIR_PRE_TOLERANCE = 0.5  # allow hit detected slightly before shot (visual leads audio)
ANALYSIS_DIR = "analysis"
LABELS_FILE = "labels.json"


# ── load & classify ─────────────────────────────────────────────────────────

def load_hits_json(video_dir):
    path = os.path.join(video_dir, HITS_FILE)
    if not os.path.exists(path):
        print(f"No {path} found. Run: python3 detect_hits.py detect --type both {video_dir}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def classify_shots(entries):
    """Pair each shot with nearest subsequent hit. Return per-video shot info."""
    from collections import OrderedDict
    by_file = OrderedDict()
    for e in entries:
        by_file.setdefault(e["file"], []).append(e)

    result = OrderedDict()
    for fname, file_entries in by_file.items():
        shots = sorted([e for e in file_entries if e.get("type") == "shot"],
                       key=lambda e: e["timestamp"])
        hits = sorted([e for e in file_entries if e.get("type") == "hit"],
                      key=lambda e: e["timestamp"])

        if not shots:
            continue

        # Pair hits to shots by closest absolute time distance.
        # Allow hits from 0.5s before to 3.0s after a shot.
        # The visual detector often fires ~0.2-0.35s before the audio
        # shot detection, so pre-shot hits are common.
        candidates = []
        for si, s in enumerate(shots):
            for hi, h in enumerate(hits):
                delay = h["timestamp"] - s["timestamp"]
                if -HIT_PAIR_PRE_TOLERANCE <= delay <= HIT_PAIR_MAX_DELAY:
                    candidates.append((abs(delay), si, hi))
        candidates.sort()

        paired_shots = set()
        paired_hits = set()
        shot_hit_map = {}
        for _, si, hi in candidates:
            if si in paired_shots or hi in paired_hits:
                continue
            shot_hit_map[si] = hi
            paired_shots.add(si)
            paired_hits.add(hi)

        shot_infos = []
        for si, s in enumerate(shots):
            if si in shot_hit_map:
                shot_infos.append({
                    "shot_ts": s["timestamp"],
                    "outcome": "hit",
                    "hit_ts": hits[shot_hit_map[si]]["timestamp"],
                    "shot_number": si + 1,
                })
            else:
                shot_infos.append({
                    "shot_ts": s["timestamp"],
                    "outcome": "miss",
                    "hit_ts": None,
                    "shot_number": si + 1,
                })

        # Add inter-shot info for doubles
        if len(shot_infos) > 1:
            for i, si in enumerate(shot_infos):
                other = shot_infos[1 - i] if len(shot_infos) == 2 else None
                si["inter_shot_delay"] = abs(shot_infos[1]["shot_ts"] - shot_infos[0]["shot_ts"]) if len(shot_infos) >= 2 else None
                si["other_outcome"] = other["outcome"] if other else None
        else:
            shot_infos[0]["inter_shot_delay"] = None
            shot_infos[0]["other_outcome"] = None

        result[fname] = shot_infos
    return result


def cmd_label(video_dir):
    """Generate labels.json with auto-detected outcomes for user review."""
    entries = load_hits_json(video_dir)
    classifications = classify_shots(entries)

    labels = []
    for fname in sorted(classifications.keys()):
        for s in classifications[fname]:
            labels.append({
                "file": fname,
                "shot": s["shot_number"],
                "timestamp": s["shot_ts"],
                "outcome": s["outcome"],
            })

    labels_path = os.path.join(video_dir, LABELS_FILE)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    total = len(labels)
    hits = sum(1 for l in labels if l["outcome"] == "hit")
    misses = total - hits

    print(f"Generated {labels_path}")
    print(f"  {total} shots: {hits} hits, {misses} misses (auto-detected)\n")
    print("Review and edit the file -- change \"outcome\" to \"hit\" or \"miss\"")
    print(f"for each shot, then run:  python3 {sys.argv[0]} {video_dir}")


def load_labels(video_dir):
    """Load labels.json if it exists, return per-video shot info."""
    labels_path = os.path.join(video_dir, LABELS_FILE)
    if not os.path.exists(labels_path):
        return None

    with open(labels_path) as f:
        labels = json.load(f)

    from collections import OrderedDict
    result = OrderedDict()
    for l in labels:
        fname = l["file"]
        result.setdefault(fname, [])
        result[fname].append({
            "shot_ts": l["timestamp"],
            "outcome": l["outcome"],
            "hit_ts": None,  # not needed for analysis
            "shot_number": l["shot"],
            "inter_shot_delay": None,
            "other_outcome": None,
        })

    # Fill in inter-shot info for doubles
    for fname, shot_infos in result.items():
        if len(shot_infos) > 1:
            for i, si in enumerate(shot_infos):
                other = shot_infos[1 - i] if len(shot_infos) == 2 else None
                si["inter_shot_delay"] = abs(shot_infos[1]["shot_ts"] - shot_infos[0]["shot_ts"])
                si["other_outcome"] = other["outcome"] if other else None

    return result


# ── per-video data extraction ───────────────────────────────────────────────

def extract_per_video_data(video_path):
    """Extract all raw signals from a video."""
    duration, fps = get_video_info(video_path)

    # Get frame dimensions
    info = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True)
    stream = json.loads(info.stdout)["streams"][0]
    orig_w, orig_h = int(stream["width"]), int(stream["height"])
    ah = int(ANALYSIS_WIDTH * orig_h / orig_w)
    ah += ah % 2

    # Extract grayscale frames
    cmd = ["ffmpeg", "-i", video_path,
           "-vf", f"scale={ANALYSIS_WIDTH}:-2,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-v", "quiet", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()

    fs = ANALYSIS_WIDTH * ah
    n = len(raw) // fs
    if n < 2:
        return None

    frames = np.frombuffer(raw[:n * fs], dtype=np.uint8).reshape(n, ah, ANALYSIS_WIDTH)
    frames_flat = frames.reshape(n, -1).astype(np.int16)

    frame_diffs = np.mean(np.abs(frames_flat[1:] - frames_flat[:-1]), axis=1)
    brightness = np.mean(frames_flat, axis=1).astype(np.float64)

    # Quadrant brightness (TL, TR, BL, BR)
    half_h, half_w = ah // 2, ANALYSIS_WIDTH // 2
    quadrants = np.zeros((n, 4), dtype=np.float64)
    quadrants[:, 0] = np.mean(frames[:, :half_h, :half_w].reshape(n, -1).astype(np.float64), axis=1)
    quadrants[:, 1] = np.mean(frames[:, :half_h, half_w:].reshape(n, -1).astype(np.float64), axis=1)
    quadrants[:, 2] = np.mean(frames[:, half_h:, :half_w].reshape(n, -1).astype(np.float64), axis=1)
    quadrants[:, 3] = np.mean(frames[:, half_h:, half_w:].reshape(n, -1).astype(np.float64), axis=1)
    quadrant_diffs = np.abs(quadrants[1:] - quadrants[:-1])

    # Audio
    audio_samples = extract_audio(video_path)
    audio_energy = compute_audio_energy(audio_samples)

    return {
        "frame_diffs": frame_diffs,
        "brightness": brightness,
        "quadrant_diffs": quadrant_diffs,
        "quadrants": quadrants,
        "audio_energy": audio_energy,
        "audio_samples": audio_samples,
        "fps": fps,
        "duration": duration,
    }


# ── per-shot feature extraction ─────────────────────────────────────────────

def extract_shot_features(video_data, shot_ts, outcome, hit_ts,
                          shot_number, inter_shot_delay, other_outcome):
    """Extract all features for one shot from pre-computed video data."""
    fps = video_data["fps"]
    diffs = video_data["frame_diffs"]
    bright = video_data["brightness"]
    qdiffs = video_data["quadrant_diffs"]
    quads = video_data["quadrants"]
    energy = video_data["audio_energy"]

    shot_frame = int(shot_ts * fps)
    pre_frames = int(WINDOW_PRE * fps)
    post_frames = int(WINDOW_POST * fps)

    f = {}
    f["shot_ts"] = shot_ts
    f["outcome"] = outcome
    f["hit_ts"] = hit_ts
    f["shot_number"] = shot_number
    f["inter_shot_delay"] = inter_shot_delay
    f["other_outcome"] = other_outcome

    # Shot-to-hit delay
    f["shot_to_hit_delay"] = (hit_ts - shot_ts) if hit_ts is not None else None

    # Diff profile (2s window)
    d_start = max(0, shot_frame - pre_frames - 1)
    d_end = min(len(diffs), shot_frame + post_frames)
    f["diff_profile"] = diffs[d_start:d_end].copy()
    f["diff_profile_offset"] = shot_frame - 1 - d_start  # index of shot in profile

    # Diff at shot
    si = min(max(shot_frame - 1, 0), len(diffs) - 1)
    f["diff_at_shot"] = float(diffs[si])
    f["diff_baseline"] = float(np.mean(diffs))
    f["diff_at_shot_ratio"] = f["diff_at_shot"] / f["diff_baseline"] if f["diff_baseline"] > 0 else 0

    # Peak diff in post-shot window
    post_start = min(si, len(diffs) - 1)
    post_end = min(si + post_frames, len(diffs))
    if post_end > post_start:
        f["peak_diff_post"] = float(np.max(diffs[post_start:post_end]))
    else:
        f["peak_diff_post"] = f["diff_at_shot"]

    # Pre-shot stability
    pre_start = max(0, si - pre_frames)
    pre_slice = diffs[pre_start:si] if si > pre_start else diffs[max(0, si-1):si+1]
    f["pre_shot_stability"] = float(np.var(pre_slice)) if len(pre_slice) > 1 else 0
    f["pre_shot_diff_mean"] = float(np.mean(pre_slice)) if len(pre_slice) > 0 else 0

    # Brightness profile
    b_start = max(0, shot_frame - pre_frames)
    b_end = min(len(bright), shot_frame + post_frames)
    f["brightness_profile"] = bright[b_start:b_end].copy()
    f["brightness_profile_offset"] = shot_frame - b_start

    # Pre-shot brightness slope
    bp_start = max(0, shot_frame - pre_frames)
    bp_slice = bright[bp_start:shot_frame]
    if len(bp_slice) > 2:
        x = np.arange(len(bp_slice))
        slope = np.polyfit(x, bp_slice, 1)[0]
        f["pre_shot_brightness_slope"] = float(slope)
    else:
        f["pre_shot_brightness_slope"] = 0.0
    f["brightness_at_shot"] = float(bright[min(shot_frame, len(bright) - 1)])

    # Post-shot recovery
    if post_end > post_start and len(diffs[post_start:post_end]) > 2:
        post_diffs = diffs[post_start:post_end]
        peak_val = np.max(post_diffs)
        baseline = f["diff_baseline"]
        half_target = (peak_val + baseline) / 2
        below = np.where(post_diffs < half_target)[0]
        f["frames_to_half_peak"] = int(below[0]) if len(below) > 0 else len(post_diffs)
        if len(post_diffs) > 2:
            x = np.arange(len(post_diffs))
            f["post_shot_recovery_rate"] = float(np.polyfit(x, post_diffs, 1)[0])
        else:
            f["post_shot_recovery_rate"] = 0.0
    else:
        f["frames_to_half_peak"] = 0
        f["post_shot_recovery_rate"] = 0.0

    # Audio features
    audio_window_rate = 1000 / AUDIO_WINDOW_MS  # windows per second
    audio_center = int(shot_ts * audio_window_rate)
    audio_pre = int(WINDOW_PRE * audio_window_rate)
    audio_post = int(WINDOW_POST * audio_window_rate)
    a_start = max(0, audio_center - audio_pre)
    a_end = min(len(energy), audio_center + audio_post)
    f["audio_profile"] = energy[a_start:a_end].copy() if a_end > a_start else np.array([])
    f["audio_profile_offset"] = audio_center - a_start

    if len(f["audio_profile"]) > 0:
        f["peak_audio_energy"] = float(np.max(f["audio_profile"]))
        # Decay rate
        peak_idx = np.argmax(f["audio_profile"])
        decay_slice = f["audio_profile"][peak_idx:peak_idx + 50]
        decay_slice = decay_slice[decay_slice > np.mean(energy) * 2]
        if len(decay_slice) > 3:
            log_e = np.log(decay_slice + 1)
            x = np.arange(len(log_e))
            f["audio_decay_rate"] = float(-np.polyfit(x, log_e, 1)[0])
        else:
            f["audio_decay_rate"] = 0.0
    else:
        f["peak_audio_energy"] = 0.0
        f["audio_decay_rate"] = 0.0

    # Quadrant motion
    q_start = max(0, shot_frame - pre_frames - 1)
    q_end = min(len(qdiffs), shot_frame + post_frames)
    q_pre_start = max(0, shot_frame - pre_frames - 1)
    q_pre_end = min(shot_frame - 1, len(qdiffs))

    if q_pre_end > q_pre_start:
        pre_qdiffs = qdiffs[q_pre_start:q_pre_end]
        # Vertical motion: top vs bottom quadrant change
        top_mean = np.mean(pre_qdiffs[:, :2], axis=1)
        bot_mean = np.mean(pre_qdiffs[:, 2:], axis=1)
        f["vertical_motion"] = float(np.mean(top_mean - bot_mean))

        # Horizontal motion: right vs left
        left_mean = np.mean(pre_qdiffs[:, [0, 2]], axis=1)
        right_mean = np.mean(pre_qdiffs[:, [1, 3]], axis=1)
        f["horizontal_motion"] = float(np.mean(right_mean - left_mean))

        f["motion_consistency"] = float(np.std(top_mean - bot_mean))
    else:
        f["vertical_motion"] = 0.0
        f["horizontal_motion"] = 0.0
        f["motion_consistency"] = 0.0

    return f


# ── statistics ──────────────────────────────────────────────────────────────

SCALAR_SIGNALS = [
    ("pre_shot_stability", "Pre-shot stability (variance)", "lower"),
    ("pre_shot_diff_mean", "Pre-shot diff mean", "lower"),
    ("pre_shot_brightness_slope", "Pre-shot brightness slope", "higher"),
    ("diff_at_shot_ratio", "Diff at shot (ratio)", "higher"),
    ("peak_diff_post", "Peak diff post-shot", "higher"),
    ("post_shot_recovery_rate", "Post-shot recovery rate", "neutral"),
    ("frames_to_half_peak", "Frames to half-peak", "neutral"),
    ("peak_audio_energy", "Peak audio energy", "neutral"),
    ("audio_decay_rate", "Audio decay rate", "neutral"),
    ("vertical_motion", "Vertical motion (pre-shot)", "neutral"),
    ("horizontal_motion", "Horizontal motion (pre-shot)", "neutral"),
    ("motion_consistency", "Motion consistency", "lower"),
]


def compute_statistics(all_features):
    hits = [f for f in all_features if f["outcome"] == "hit"]
    misses = [f for f in all_features if f["outcome"] == "miss"]

    stats = {}
    for key, label, direction in SCALAR_SIGNALS:
        h_vals = np.array([f[key] for f in hits if f[key] is not None], dtype=np.float64)
        m_vals = np.array([f[key] for f in misses if f[key] is not None], dtype=np.float64)

        entry = {"label": label, "direction": direction}
        entry["hit_mean"] = float(np.mean(h_vals)) if len(h_vals) > 0 else 0
        entry["hit_std"] = float(np.std(h_vals)) if len(h_vals) > 0 else 0
        entry["miss_mean"] = float(np.mean(m_vals)) if len(m_vals) > 0 else 0
        entry["miss_std"] = float(np.std(m_vals)) if len(m_vals) > 0 else 0

        # Cohen's d
        if len(h_vals) > 1 and len(m_vals) > 1:
            pooled = np.sqrt(((len(h_vals)-1)*np.var(h_vals) + (len(m_vals)-1)*np.var(m_vals))
                             / (len(h_vals) + len(m_vals) - 2))
            entry["cohens_d"] = float((entry["hit_mean"] - entry["miss_mean"]) / pooled) if pooled > 0 else 0
        else:
            entry["cohens_d"] = 0

        entry["hit_values"] = h_vals
        entry["miss_values"] = m_vals
        stats[key] = entry

    return stats


def compute_per_miss_notes(misses, stats):
    notes = []
    for f in misses:
        deviations = []
        for key, label, direction in SCALAR_SIGNALS:
            s = stats[key]
            if s["hit_std"] > 0 and f[key] is not None:
                z = (f[key] - s["hit_mean"]) / s["hit_std"]
                if abs(z) > 1.0:
                    deviations.append((abs(z), label, f[key], s["hit_mean"], z))
        deviations.sort(reverse=True)
        note_parts = []
        for _, label, val, hit_mean, z in deviations[:3]:
            direction = "above" if z > 0 else "below"
            note_parts.append(f"{label}: {val:.2f} ({z:+.1f} std {direction} hit avg {hit_mean:.2f})")
        notes.append((f["video"], f["shot_ts"], note_parts))
    return notes


# ── text report ─────────────────────────────────────────────────────────────

def print_report(all_features, stats, per_miss_notes, file=None):
    def out(text=""):
        print(text)
        if file:
            file.write(text + "\n")

    hits = [f for f in all_features if f["outcome"] == "hit"]
    misses = [f for f in all_features if f["outcome"] == "miss"]
    total = len(all_features)
    hit_rate = len(hits) / total * 100 if total > 0 else 0

    out("=" * 72)
    out("               CLAY TARGET SHOOTING ANALYSIS")
    out("=" * 72)
    out(f"\n  Videos analyzed: {len(set(f['video'] for f in all_features))}")
    out(f"  Total shots:     {total}")
    out(f"  Hits:            {len(hits)}  ({hit_rate:.1f}%)")
    out(f"  Misses:          {len(misses)}  ({100-hit_rate:.1f}%)")

    # Hit delay stats
    delays = [f["shot_to_hit_delay"] for f in hits if f["shot_to_hit_delay"] is not None]
    if delays:
        out(f"\n  Avg shot-to-hit delay: {np.mean(delays):.2f}s (std {np.std(delays):.2f}s)")

    out(f"\n{'='*72}")
    out("SIGNAL COMPARISON: HITS vs MISSES")
    out("=" * 72)

    header = f"{'Signal':<32} {'Hits':>18} {'Misses':>18} {'d':>8}"
    out(header)
    out("-" * len(header))

    ranked = sorted(stats.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
    for key, s in ranked:
        effect = ""
        ad = abs(s["cohens_d"])
        if ad > 0.8:
            effect = "***"
        elif ad > 0.5:
            effect = "**"
        elif ad > 0.2:
            effect = "*"
        h_str = f"{s['hit_mean']:.2f}+/-{s['hit_std']:.2f}"
        m_str = f"{s['miss_mean']:.2f}+/-{s['miss_std']:.2f}"
        out(f"  {s['label']:<30} {h_str:>18} {m_str:>18} {s['cohens_d']:>+7.2f} {effect}")

    out(f"\n  Effect size: |d|>0.8=*** 0.5-0.8=** 0.2-0.5=* <0.2=(ns)")

    # Top factors
    out(f"\n{'='*72}")
    out("TOP DISTINGUISHING FACTORS")
    out("=" * 72)
    for i, (key, s) in enumerate(ranked[:5]):
        if abs(s["cohens_d"]) < 0.1:
            break
        direction = "higher" if s["cohens_d"] > 0 else "lower"
        pct = abs(s["hit_mean"] - s["miss_mean"]) / abs(s["hit_mean"]) * 100 if s["hit_mean"] != 0 else 0
        out(f"\n  {i+1}. {s['label']} (d={s['cohens_d']:+.2f})")
        out(f"     Hits avg: {s['hit_mean']:.3f}  |  Misses avg: {s['miss_mean']:.3f}")
        out(f"     Hits are {direction} by {pct:.0f}%")

    # Per-miss analysis
    if per_miss_notes:
        out(f"\n{'='*72}")
        out("PER-MISS ANALYSIS")
        out("=" * 72)
        for i, (video, ts, notes) in enumerate(per_miss_notes):
            out(f"\n  {video} @ {ts}s (miss #{i+1} of {len(per_miss_notes)})")
            if notes:
                for note in notes:
                    out(f"    - {note}")
            else:
                out(f"    - No significant deviations from hit averages")

    out()


# ── plots ───────────────────────────────────────────────────────────────────

def generate_plots(all_features, stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    hits = [f for f in all_features if f["outcome"] == "hit"]
    misses = [f for f in all_features if f["outcome"] == "miss"]

    plot_profiles(hits, misses, "diff_profile", "diff_profile_offset",
                  "Frame Difference Profiles", "Mean Abs Diff",
                  os.path.join(output_dir, "diff_profiles.png"))

    plot_profiles(hits, misses, "brightness_profile", "brightness_profile_offset",
                  "Brightness Profiles", "Mean Brightness",
                  os.path.join(output_dir, "brightness_profiles.png"))

    plot_profiles(hits, misses, "audio_profile", "audio_profile_offset",
                  "Audio Energy Profiles", "RMS Energy",
                  os.path.join(output_dir, "audio_profiles.png"),
                  time_scale=AUDIO_WINDOW_MS / 1000)

    plot_distributions(stats, output_dir)
    plot_effect_sizes(stats, output_dir)

    if misses:
        plot_quadrant_motion(hits, misses, output_dir)

    print(f"  Plots saved to {output_dir}/")


def plot_profiles(hits, misses, profile_key, offset_key, title, ylabel, save_path,
                  time_scale=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    for group, color, label in [(hits, "green", "Hits"), (misses, "red", "Misses")]:
        if not group:
            continue
        profiles = []
        for f in group:
            p = f[profile_key]
            offset = f[offset_key]
            if len(p) == 0:
                continue
            if time_scale is None:
                fps = 40  # approximate
                t = (np.arange(len(p)) - offset) / fps
            else:
                t = (np.arange(len(p)) - offset) * time_scale
            ax.plot(t, p, color=color, alpha=0.15, linewidth=0.8)
            profiles.append((t, p))

        # Mean profile (interpolated to common time axis)
        if profiles:
            common_t = np.linspace(-WINDOW_PRE, WINDOW_POST, 200)
            interpolated = []
            for t, p in profiles:
                if len(t) > 1:
                    interp = np.interp(common_t, t, p, left=np.nan, right=np.nan)
                    interpolated.append(interp)
            if interpolated:
                stacked = np.array(interpolated)
                mean_p = np.nanmean(stacked, axis=0)
                ax.plot(common_t, mean_p, color=color, linewidth=2.5, label=f"{label} (n={len(group)})")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Shot")
    ax.set_xlabel("Time relative to shot (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distributions(stats, output_dir):
    signals = [(k, s) for k, s in stats.items()
               if len(s["hit_values"]) > 0 or len(s["miss_values"]) > 0]
    n = len(signals)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (key, s) in enumerate(signals):
        ax = axes[i]
        data = []
        labels = []
        colors = []
        if len(s["hit_values"]) > 0:
            data.append(s["hit_values"])
            labels.append("Hits")
            colors.append("green")
        if len(s["miss_values"]) > 0:
            data.append(s["miss_values"])
            labels.append("Misses")
            colors.append("red")

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.3)

        # Scatter individual points
        for j, (d, c) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(42).normal(0, 0.04, len(d))
            ax.scatter(np.full(len(d), j + 1) + jitter, d, color=c, alpha=0.5, s=15, zorder=3)

        ax.set_title(s["label"], fontsize=9)
        ad = abs(s["cohens_d"])
        stars = "***" if ad > 0.8 else "**" if ad > 0.5 else "*" if ad > 0.2 else ""
        ax.set_xlabel(f"d={s['cohens_d']:+.2f} {stars}", fontsize=8)

    for i in range(len(signals), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Signal Distributions: Hits vs Misses", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "signal_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effect_sizes(stats, output_dir):
    items = sorted(stats.items(), key=lambda x: abs(x[1]["cohens_d"]))
    labels = [s["label"] for _, s in items]
    values = [s["cohens_d"] for _, s in items]
    colors = ["green" if v > 0 else "red" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    ax.barh(labels, values, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    for x in [-0.8, -0.5, 0.5, 0.8]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Cohen's d (green = higher in hits, red = higher in misses)")
    ax.set_title("Effect Sizes: Hit vs Miss Signal Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "effect_sizes.png"), dpi=150)
    plt.close(fig)


def plot_quadrant_motion(hits, misses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))

    for group, color, marker, label in [(hits, "green", "o", "Hits"),
                                         (misses, "red", "x", "Misses")]:
        v = [f["vertical_motion"] for f in group]
        h = [f["horizontal_motion"] for f in group]
        ax.scatter(h, v, color=color, marker=marker, s=60, alpha=0.6, label=label)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Horizontal motion (right - left)")
    ax.set_ylabel("Vertical motion (up - down)")
    ax.set_title("Pre-shot Camera Motion Direction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quadrant_motion.png"), dpi=150)
    plt.close(fig)


# ── miss frame extraction ──────────────────────────────────────────────────

def extract_miss_frames(misses, stats, video_dir, output_dir):
    if not misses:
        return

    os.makedirs(output_dir, exist_ok=True)

    for f in misses:
        vpath = os.path.join(video_dir, f["video"])
        if not os.path.exists(vpath):
            continue

        # Extract frame via FFmpeg
        cmd = ["ffmpeg", "-ss", f"{f['shot_ts']:.3f}", "-i", vpath,
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg",
               "-v", "quiet", "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if not proc.stdout:
            continue

        img = Image.open(io.BytesIO(proc.stdout))

        # Annotate
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
            font_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_sm = font

        # Red MISS banner
        draw.rectangle([(0, 0), (img.width, 70)], fill=(180, 0, 0, 200))
        draw.text((20, 10), f"MISS - {f['video']} @ {f['shot_ts']}s",
                  fill="white", font=font)

        # Feature notes
        y = 80
        for key, label, _ in SCALAR_SIGNALS[:6]:
            s = stats[key]
            val = f[key]
            if val is not None and s["hit_std"] > 0:
                z = (val - s["hit_mean"]) / s["hit_std"]
                text = f"{label}: {val:.2f} (hit avg: {s['hit_mean']:.2f}, z={z:+.1f})"
                for dx, dy in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    draw.text((20 + dx, y + dy), text, fill="black", font=font_sm)
                draw.text((20, y), text, fill="white", font=font_sm)
                y += 35

        fname = f"miss_{f['video'].replace('.MP4', '')}_{f['shot_ts']}s.jpg"
        img.save(os.path.join(output_dir, fname), quality=90)

    print(f"  Miss frames saved to {output_dir}/")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze hit vs miss patterns in clay target shooting videos.",
        epilog="Workflow:\n"
               "  1. python3 detect_hits.py detect --type both /path/to/videos\n"
               "  2. python3 analyze_shots.py label /path/to/videos\n"
               "     (edit labels.json to correct any misclassifications)\n"
               "  3. python3 analyze_shots.py /path/to/videos\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default=None,
                        help="label (generate labels.json) or omit to analyze")
    parser.add_argument("video_dir", nargs="?", default=".",
                        help="Directory with SHOT*.MP4 and hits.json (default: .)")
    args = parser.parse_args()

    # Handle ambiguity: if command looks like a path, treat as video_dir
    if args.command and args.command != "label":
        args.video_dir = args.command
        args.command = None

    video_dir = os.path.abspath(args.video_dir)

    if args.command == "label":
        cmd_label(video_dir)
        return

    # Load classifications: use labels.json if it exists, otherwise auto-detect
    labels_path = os.path.join(video_dir, LABELS_FILE)
    classifications = load_labels(video_dir)
    if classifications is not None:
        print(f"Using {labels_path} for classifications")
    else:
        print(f"No {LABELS_FILE} found, using auto-detection")
        print(f"  (Run: python3 {sys.argv[0]} label {video_dir}  to generate editable labels)\n")
        entries = load_hits_json(video_dir)
        classifications = classify_shots(entries)

    total_shots = sum(len(v) for v in classifications.values())
    total_hits = sum(1 for v in classifications.values() for s in v if s["outcome"] == "hit")
    total_misses = total_shots - total_hits
    print(f"Classified {total_shots} shots: {total_hits} hits, {total_misses} misses\n")

    if total_shots == 0:
        print("No shots found. Run: python3 detect_hits.py detect --type both")
        sys.exit(1)

    # Extract features
    all_features = []
    video_files = sorted(classifications.keys())
    for i, vname in enumerate(video_files):
        sys.stdout.write(f"[{i+1}/{len(video_files)}] Analyzing {vname}...")
        sys.stdout.flush()
        vpath = os.path.join(video_dir, vname)
        if not os.path.exists(vpath):
            print(" NOT FOUND, skipping")
            continue

        data = extract_per_video_data(vpath)
        if data is None:
            print(" too short, skipping")
            continue

        for shot_info in classifications[vname]:
            features = extract_shot_features(data, **shot_info)
            features["video"] = vname
            all_features.append(features)
        print(" done")

    hits = [f for f in all_features if f["outcome"] == "hit"]
    misses = [f for f in all_features if f["outcome"] == "miss"]

    if not hits:
        print("\nNo hits detected. Cannot compare.")
        sys.exit(1)

    # Statistics
    stats = compute_statistics(all_features)
    per_miss_notes = compute_per_miss_notes(misses, stats)

    # Report
    output_dir = os.path.join(video_dir, ANALYSIS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.txt")
    print()
    with open(report_path, "w") as report_file:
        print_report(all_features, stats, per_miss_notes, file=report_file)
    print(f"  Report saved to {report_path}")

    # Plots and frames
    if misses:
        generate_plots(all_features, stats, output_dir)
        extract_miss_frames(misses, stats, video_dir, output_dir)
    else:
        print("No misses found -- all shots hit! No comparison to make.")
        print("Generating hit-only plots...")
        generate_plots(all_features, stats, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
