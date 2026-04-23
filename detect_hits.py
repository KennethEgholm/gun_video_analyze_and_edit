#!/usr/bin/env python3
"""
Detect clay target hits in shotgun-camera videos and compile a highlight reel.

Usage:
    python3 detect_hits.py detect [VIDEO_DIR]    Analyze videos, write hits.json
    python3 detect_hits.py compile [VIDEO_DIR]   Read hits.json, create compilation
    python3 detect_hits.py [VIDEO_DIR]           Do both (detect + compile)

The detect step writes a hits.json file that you can review and edit
(e.g. remove false positives) before running compile.

VIDEO_DIR defaults to the current directory.

Detection strategies:
1. Visual hit detection: Compute frame-to-frame differences on downscaled
   grayscale frames. Find spikes (> mean + 3*std) as candidates. Filter
   using brightness: reject spikes where the gun is lowered toward the
   ground (frame brightness drops well below average).
2. Audio shot detection: Extract PCM audio, compute 10ms-window RMS energy
   envelope. Find spikes (> mean + 5*std) as gunshot transients.
"""

import argparse
import shutil
import subprocess
import sys
import os
import json
import tempfile
import glob
import numpy as np

HITS_FILE = "hits.json"
OUTPUT_FILE = "hits_compilation.mp4"
CLIP_DURATION = 1.5
CLIP_PRE = 0.5         # seconds before the hit
ANALYSIS_WIDTH = 320
SPIKE_THRESHOLD_MULTIPLIER = 3.0
MIN_SPIKE_SEPARATION_SEC = 1.0
BRIGHTNESS_DROP_MAX = 0.75

# Audio gunshot detection
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_MS = 10                # 10ms energy windows
AUDIO_WINDOW_SAMPLES = 160          # AUDIO_SAMPLE_RATE * AUDIO_WINDOW_MS / 1000
AUDIO_SPIKE_MULTIPLIER = 5.0        # energy > mean + 5*std
AUDIO_MIN_SPIKE_SEPARATION_SEC = 0.5
AUDIO_MIN_PEAK_RMS = 18000          # reject transients below this RMS (e.g. gun break-open)
MERGE_MAX_GAP_SEC = 5.0             # max gap between entries before splitting a merged clip


def check_dependencies():
    """Verify required external tools are installed; exit with a friendly
    message if not."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        print(f"Error: required tool(s) not found on PATH: {', '.join(missing)}")
        print("Install FFmpeg, e.g. on macOS:  brew install ffmpeg")
        sys.exit(1)


def get_video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(f"Error: ffprobe failed to read {path}")
        if result.stderr:
            print(result.stderr.strip())
        sys.exit(1)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    stream = data["streams"][0]
    fps_parts = stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])
    return duration, fps


def analyze_video(path, analysis_width=ANALYSIS_WIDTH):
    """Compute per-frame diffs and brightness."""
    duration, fps = get_video_info(path)

    cmd = [
        "ffmpeg", "-i", path,
        "-vf", f"scale={analysis_width}:-2,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-v", "quiet", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()

    # Derive actual height from output size
    frame_pixels = len(raw) // (analysis_width * max(1, len(raw) // (analysis_width * 2)))
    # More robust: get actual dimensions
    info = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True
    )
    stream = json.loads(info.stdout)["streams"][0]
    orig_w, orig_h = int(stream["width"]), int(stream["height"])
    ah = int(analysis_width * orig_h / orig_w)
    ah += ah % 2

    fs = analysis_width * ah
    n = len(raw) // fs
    if n < 2:
        return np.array([]), np.array([]), fps, duration

    frames = np.frombuffer(raw[:n * fs], dtype=np.uint8).reshape(n, -1).astype(np.int16)
    full_diffs = np.mean(np.abs(frames[1:] - frames[:-1]), axis=1)
    brightness = np.mean(frames, axis=1)

    return full_diffs, brightness, fps, duration


def find_hits(full_diffs, brightness, fps):
    """
    Find diff spikes, reject when the gun is clearly lowered.
    Returns list of (timestamp, label, reason) tuples.
    """
    if len(full_diffs) == 0:
        return []

    mean_val = np.mean(full_diffs)
    std_val = np.std(full_diffs)
    if std_val < 0.5:
        return []

    threshold = mean_val + SPIKE_THRESHOLD_MULTIPLIER * std_val
    spike_indices = np.where(full_diffs > threshold)[0]
    if len(spike_indices) == 0:
        return []

    min_sep_frames = int(MIN_SPIKE_SEPARATION_SEC * fps)
    groups = []
    cur = [spike_indices[0]]
    for i in range(1, len(spike_indices)):
        if spike_indices[i] - cur[-1] <= min_sep_frames:
            cur.append(spike_indices[i])
        else:
            groups.append(cur)
            cur = [spike_indices[i]]
    groups.append(cur)

    avg_brightness = np.mean(brightness)
    hits = []

    for group in groups:
        group_arr = np.array(group)
        peak_idx = group_arr[np.argmax(full_diffs[group_arr])]
        bright_at_peak = brightness[min(peak_idx + 1, len(brightness) - 1)]
        bright_ratio = bright_at_peak / avg_brightness
        timestamp = round((peak_idx + 1) / fps, 2)

        reason = f"bright={bright_ratio:.2f}"
        if bright_ratio >= BRIGHTNESS_DROP_MAX:
            hits.append((timestamp, "HIT", reason))
        else:
            hits.append((timestamp, "REJECT", reason))

    return hits


def extract_audio(path, sample_rate=AUDIO_SAMPLE_RATE):
    """Extract raw PCM audio from video file. Returns int16 numpy array."""
    cmd = [
        "ffmpeg", "-i", path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-v", "quiet", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()

    if len(raw) < 2:
        return np.array([], dtype=np.int16)
    return np.frombuffer(raw, dtype=np.int16)


def compute_audio_energy(samples, window_samples=AUDIO_WINDOW_SAMPLES):
    """Compute RMS energy in non-overlapping windows. Returns float64 array."""
    n_windows = len(samples) // window_samples
    if n_windows == 0:
        return np.array([])
    trimmed = samples[:n_windows * window_samples].reshape(n_windows, window_samples)
    return np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))


def find_gunshots(energy, samples=None, window_ms=AUDIO_WINDOW_MS,
                  sample_rate=AUDIO_SAMPLE_RATE):
    """
    Find gunshot transients in RMS energy envelope.
    Returns list of (timestamp, label, reason) tuples.
    """
    if len(energy) == 0:
        return []

    mean_val = np.mean(energy)
    std_val = np.std(energy)
    if std_val < 1.0:
        return []

    threshold = mean_val + AUDIO_SPIKE_MULTIPLIER * std_val
    spike_indices = np.where(energy > threshold)[0]
    if len(spike_indices) == 0:
        return []

    min_sep_windows = int(AUDIO_MIN_SPIKE_SEPARATION_SEC * 1000 / window_ms)
    groups = []
    cur = [spike_indices[0]]
    for i in range(1, len(spike_indices)):
        if spike_indices[i] - cur[-1] <= min_sep_windows:
            cur.append(spike_indices[i])
        else:
            groups.append(cur)
            cur = [spike_indices[i]]
    groups.append(cur)

    results = []
    for group in groups:
        group_arr = np.array(group)
        peak_idx = group_arr[np.argmax(energy[group_arr])]
        timestamp = round(peak_idx * window_ms / 1000.0, 2)
        ratio = energy[peak_idx] / mean_val if mean_val > 0 else 0

        # Reject if peak RMS is too low (e.g. gun break-open, not a gunshot)
        if samples is not None:
            center = int(timestamp * sample_rate)
            half = int(sample_rate * 0.05)  # 100ms window
            chunk = samples[max(0, center - half):center + half]
            peak_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            if peak_rms < AUDIO_MIN_PEAK_RMS:
                continue

        results.append((timestamp, "SHOT", f"energy_ratio={ratio:.1f}"))

    return results


# -- detect command -----------------------------------------------------------

def cmd_detect(video_dir, detect_type="both"):
    if not os.path.isdir(video_dir):
        print(f"Error: {video_dir} is not a directory.")
        sys.exit(1)

    video_files = sorted(
        glob.glob(os.path.join(video_dir, "SHOT*.MP4"))
        + glob.glob(os.path.join(video_dir, "SHOT*.mp4"))
    )
    if not video_files:
        print(f"No SHOT*.MP4 files found in {video_dir}")
        print("Hint: filenames must start with 'SHOT' and end in .MP4 / .mp4.")
        sys.exit(1)

    do_hits = detect_type in ("hit", "both")
    do_shots = detect_type in ("shot", "both")
    print(f"Analyzing {len(video_files)} videos "
          f"(detecting: {detect_type})...\n")

    results = []
    total_hits = 0
    total_shots = 0
    total_rejected = 0

    for i, vpath in enumerate(video_files):
        vname = os.path.basename(vpath)
        sys.stdout.write(f"[{i+1}/{len(video_files)}] {vname}")
        sys.stdout.flush()

        hits = []
        rejects = []
        shots = []

        if do_hits:
            full, bright, fps, duration = analyze_video(vpath)
            detections = find_hits(full, bright, fps)
            hits = [(t, r) for t, label, r in detections if label == "HIT"]
            rejects = [(t, r) for t, label, r in detections if label == "REJECT"]

        if do_shots:
            audio_samples = extract_audio(vpath)
            energy = compute_audio_energy(audio_samples)
            audio_detections = find_gunshots(energy, samples=audio_samples)
            shots = [(t, r) for t, label, r in audio_detections if label == "SHOT"]

        parts = []
        if shots:
            parts.append(f"{len(shots)} shot(s) at {[f'{t:.1f}s' for t, _ in shots]}")
        if hits:
            parts.append(f"{len(hits)} hit(s) at {[f'{t:.1f}s' for t, _ in hits]}")
        if rejects:
            parts.append(f"{len(rejects)} rejected: {[f'{t:.1f}s ({r})' for t, r in rejects]}")
        if parts:
            print(f" -> {'; '.join(parts)}")
        else:
            print(" -> no activity")

        total_hits += len(hits)
        total_shots += len(shots)
        total_rejected += len(rejects)

        for ts, _ in shots:
            results.append({"file": vname, "timestamp": ts, "type": "shot"})
        for ts, _ in hits:
            results.append({"file": vname, "timestamp": ts, "type": "hit"})

    results.sort(key=lambda r: (r["file"], r["timestamp"]))

    hits_path = os.path.join(video_dir, HITS_FILE)
    with open(hits_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Shots: {total_shots}  |  Hits: {total_hits}  |  Rejected: {total_rejected}")
    print(f"Written to {hits_path}")
    print()
    print("Next steps:")
    print(f"  - Review/edit {HITS_FILE} to remove false positives, then:")
    print(f"      python3 {sys.argv[0]} compile {video_dir}    # build highlight reel")
    if total_shots > 0:
        print(f"      python3 analyze_shots.py label {video_dir}   # hit/miss analysis")


def _generate_sfx(path, sample_rate=44100):
    """Generate a punchy video-game-style shotgun sound effect as WAV."""
    import wave

    duration = 0.5
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)

    # Sharp noise burst (attack)
    noise = np.random.default_rng(42).standard_normal(n)
    attack = noise * np.exp(-t * 60) * 0.7

    # Low-frequency boom (sine sweep 180Hz -> 40Hz)
    freq = 180 * np.exp(-t * 4) + 40
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    boom = np.sin(phase) * np.exp(-t * 5) * 1.0

    # Mid-range crack
    crack = noise * np.exp(-t * 35) * 0.5

    # Pump-action tail (short metallic click at ~80ms)
    pump_t = np.maximum(t - 0.08, 0)
    pump = noise * np.exp(-pump_t * 80) * np.where(t > 0.08, 0.3, 0)

    signal = attack + boom + crack + pump
    signal = signal / np.max(np.abs(signal)) * 0.9
    signal_int = (signal * 32767).astype(np.int16)

    with wave.open(path, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(signal_int.tobytes())


def _make_overlay(filename, start_sec, out_path, width=1920, height=1080):
    """Create a transparent PNG with filename and timecode text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36)
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = font_large

    m, s = divmod(start_sec, 60)
    h, m = divmod(int(m), 60)
    timecode = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((20 + dx, 20 + dy), filename, fill=(0, 0, 0, 200), font=font_large)
        draw.text((20 + dx, 64 + dy), timecode, fill=(0, 0, 0, 200), font=font_small)
    draw.text((20, 20), filename, fill=(255, 255, 255, 255), font=font_large)
    draw.text((20, 64), timecode, fill=(255, 255, 255, 255), font=font_small)

    img.save(out_path)


# -- compile command ----------------------------------------------------------

def cmd_compile(video_dir, merge=False, overlay=False, sfx=False):
    hits_path = os.path.join(video_dir, HITS_FILE)
    if not os.path.exists(hits_path):
        print(f"No {hits_path} found. Run detect first:")
        print(f"  python3 {sys.argv[0]} detect")
        sys.exit(1)

    with open(hits_path) as f:
        hits = json.load(f)

    if not hits:
        print("No entries in JSON file.")
        sys.exit(1)

    # Build clip list: either one per entry, or one per file (--merge)
    if merge:
        # Group entries by file, then split on gaps > MERGE_MAX_GAP_SEC
        from collections import OrderedDict
        by_file = OrderedDict()
        for entry in hits:
            by_file.setdefault(entry["file"], []).append(entry)
        clips_to_extract = []
        for fname, entries in by_file.items():
            timestamps = sorted(e["timestamp"] for e in entries)
            groups = [[timestamps[0]]]
            for ts in timestamps[1:]:
                if ts - groups[-1][-1] > MERGE_MAX_GAP_SEC:
                    groups.append([ts])
                else:
                    groups[-1].append(ts)
            for group in groups:
                clips_to_extract.append({
                    "file": fname,
                    "start_ts": group[0],
                    "end_ts": group[-1],
                    "count": len(group),
                    "timestamps": group,
                })
        print(f"Compiling {len(clips_to_extract)} merged clips "
              f"({len(hits)} entries)...\n")
    else:
        clips_to_extract = [
            {"file": e["file"], "start_ts": e["timestamp"],
             "end_ts": e["timestamp"], "count": 1,
             "type": e.get("type", "hit"),
             "timestamps": [e["timestamp"]]}
            for e in hits
        ]
        print(f"Compiling {len(hits)} clips...\n")

    clip_dir = tempfile.mkdtemp(prefix="clay_hits_")
    clip_paths = []

    sfx_path = None
    if sfx:
        sfx_path = os.path.join(clip_dir, "sfx.wav")
        _generate_sfx(sfx_path)

    for i, clip in enumerate(clips_to_extract):
        vpath = os.path.join(video_dir, clip["file"])

        if not os.path.exists(vpath):
            print(f"  WARNING: {clip['file']} not found, skipping")
            continue

        duration, _ = get_video_info(vpath)
        start = max(0, clip["start_ts"] - CLIP_PRE)
        clip_len = (clip["end_ts"] - clip["start_ts"]) + CLIP_DURATION
        if start + clip_len > duration:
            clip_len = duration - start

        clip_path = os.path.join(clip_dir, f"clip_{i:04d}.mp4")
        vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2")

        overlay_path = None
        if overlay:
            overlay_path = os.path.join(clip_dir, f"overlay_{i:04d}.png")
            _make_overlay(clip["file"], start, overlay_path)

        # Build FFmpeg command with optional overlay and SFX
        inputs = ["-ss", f"{start:.3f}", "-i", vpath]
        input_idx = 1
        overlay_idx = sfx_idx = None

        if overlay_path:
            inputs += ["-i", overlay_path]
            overlay_idx = input_idx
            input_idx += 1

        if sfx_path:
            inputs += ["-i", sfx_path]
            sfx_idx = input_idx
            input_idx += 1

        use_filter_complex = overlay_path or sfx_path
        filter_parts = []
        vout = "[0:v]"
        aout = "[0:a]"

        if use_filter_complex:
            filter_parts.append(f"[0:v]{vf}[scaled]")
            vout = "[scaled]"

            if overlay_path:
                filter_parts.append(f"{vout}[{overlay_idx}:v]overlay=0:0[vout]")
                vout = "[vout]"

            if sfx_path:
                offsets = [max(0, ts - start) for ts in clip["timestamps"]]
                n = len(offsets)
                if n == 1:
                    delay_ms = int(offsets[0] * 1000)
                    filter_parts.append(
                        f"[{sfx_idx}:a]adelay={delay_ms}|{delay_ms}[sfx]")
                    filter_parts.append(
                        f"[0:a][sfx]amix=inputs=2:duration=first"
                        f":dropout_transition=0[aout]")
                else:
                    filter_parts.append(
                        f"[{sfx_idx}:a]asplit={n}"
                        + "".join(f"[s{j}]" for j in range(n)))
                    for j, off in enumerate(offsets):
                        delay_ms = int(off * 1000)
                        filter_parts.append(
                            f"[s{j}]adelay={delay_ms}|{delay_ms}[d{j}]")
                    mix_in = "[0:a]" + "".join(f"[d{j}]" for j in range(n))
                    filter_parts.append(
                        f"{mix_in}amix=inputs={n + 1}:duration=first"
                        f":dropout_transition=0[aout]")
                aout = "[aout]"

        if use_filter_complex:
            fc = ";".join(filter_parts)
            maps = ["-map", vout, "-map", aout]
            cmd = (["ffmpeg", "-y"] + inputs
                   + ["-t", f"{clip_len:.3f}",
                      "-filter_complex", fc]
                   + maps
                   + ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                      "-c:a", "aac", "-b:a", "128k",
                      "-r", "30", "-v", "quiet", clip_path])
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", vpath,
                "-t", f"{clip_len:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k",
                "-vf", vf,
                "-r", "30", "-v", "quiet",
                clip_path
            ]
        subprocess.run(cmd, check=True)
        clip_paths.append(clip_path)

        total = len(clips_to_extract)
        if merge:
            print(f"  [{i+1}/{total}] {clip['file']} "
                  f"({clip['count']} entries, {clip_len:.1f}s)")
        else:
            print(f"  [{i+1}/{total}] {clip['file']} "
                  f"@ {clip['start_ts']}s ({clip.get('type', 'hit')})")

    if not clip_paths:
        print("No clips extracted.")
        sys.exit(1)

    output_path = os.path.join(video_dir, OUTPUT_FILE)
    concat_list = os.path.join(clip_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", "-v", "quiet",
        output_path
    ], check=True)

    import shutil
    shutil.rmtree(clip_dir)

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", output_path],
        capture_output=True, text=True
    )
    result_dur = float(json.loads(result.stdout)["format"]["duration"])
    print(f"\nDone! {output_path}")
    print(f"Duration: {result_dur:.1f}s ({len(clip_paths)} clips x ~{CLIP_DURATION}s)")


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect clay target hits and compile a highlight reel.",
        epilog="Examples:\n"
               "  python3 detect_hits.py detect ~/videos\n"
               "  python3 detect_hits.py compile ~/videos\n"
               "  python3 detect_hits.py ~/videos\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        help="detect, compile, or omit for both",
    )
    parser.add_argument(
        "video_dir", nargs="?", default=".",
        help="directory containing SHOT*.MP4 files (default: current dir)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="merge multiple detections from the same file into one clip",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="burn filename and source timecode onto each clip",
    )
    parser.add_argument(
        "--type", choices=["shot", "hit", "both"], default="both",
        help="detection type: shot (audio), hit (visual), or both (default)",
    )
    parser.add_argument(
        "--sfx", action="store_true",
        help="replace gunshot audio with a video-game-style sound effect",
    )
    args = parser.parse_args()

    # Handle ambiguity: if command looks like a path, treat it as video_dir
    if args.command and args.command not in ("detect", "compile"):
        args.video_dir = args.command
        args.command = None

    check_dependencies()
    video_dir = os.path.abspath(args.video_dir)

    if args.command == "detect":
        cmd_detect(video_dir, detect_type=args.type)
    elif args.command == "compile":
        cmd_compile(video_dir, merge=args.merge, overlay=args.overlay,
                    sfx=args.sfx)
    else:
        cmd_detect(video_dir, detect_type=args.type)
        print()
        cmd_compile(video_dir, merge=args.merge, overlay=args.overlay,
                    sfx=args.sfx)


if __name__ == "__main__":
    main()
