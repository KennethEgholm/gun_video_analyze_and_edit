#!/usr/bin/env python3
"""
Detect gunshots in shotgun-camera videos and compile a highlight reel.

Usage:
    python3 detect_shots.py detect [VIDEO_DIR]    Analyze videos, write shots.json
    python3 detect_shots.py compile [VIDEO_DIR]   Read shots.json, create compilation
    python3 detect_shots.py [VIDEO_DIR]           Do both (detect + compile)

The detect step writes a shots.json file that you can review and edit
(e.g. remove false positives) before running compile.

VIDEO_DIR defaults to the current directory.
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

SHOTS_FILE = "shots.json"
OUTPUT_FILE = "shots_compilation.mp4"
CLIP_DURATION = 1.5
CLIP_PRE = 0.5

# Audio gunshot detection
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_MS = 10
AUDIO_WINDOW_SAMPLES = 160          # AUDIO_SAMPLE_RATE * AUDIO_WINDOW_MS / 1000
AUDIO_SPIKE_MULTIPLIER = 5.0        # energy > mean + 5*std
AUDIO_MIN_SPIKE_SEPARATION_SEC = 0.5
AUDIO_MIN_PEAK_RMS = 18000          # reject transients below this RMS (e.g. gun break-open)
MERGE_MAX_GAP_SEC = 5.0             # max gap between entries before splitting a merged clip


def check_dependencies():
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
    """Find gunshot transients in RMS energy envelope. Returns list of timestamps (seconds)."""
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

        # Reject if peak RMS is too low (e.g. gun break-open, not a gunshot)
        if samples is not None:
            center = int(timestamp * sample_rate)
            half = int(sample_rate * 0.05)
            chunk = samples[max(0, center - half):center + half]
            peak_rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
            if peak_rms < AUDIO_MIN_PEAK_RMS:
                continue

        results.append(timestamp)

    return results


# -- detect command -----------------------------------------------------------

def cmd_detect(video_dir):
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

    print(f"Analyzing {len(video_files)} videos...\n")

    results = []
    total_shots = 0

    for i, vpath in enumerate(video_files):
        vname = os.path.basename(vpath)
        sys.stdout.write(f"[{i+1}/{len(video_files)}] {vname}")
        sys.stdout.flush()

        audio_samples = extract_audio(vpath)
        energy = compute_audio_energy(audio_samples)
        shots = find_gunshots(energy, samples=audio_samples)

        if shots:
            print(f" -> {len(shots)} shot(s) at "
                  f"{[f'{t:.1f}s' for t in shots]}")
        else:
            print(" -> no shots")

        total_shots += len(shots)
        for shot_ts in shots:
            results.append({"file": vname, "timestamp": shot_ts})

    results.sort(key=lambda r: (r["file"], r["timestamp"]))

    shots_path = os.path.join(video_dir, SHOTS_FILE)
    with open(shots_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Shots: {total_shots}")
    print(f"Written to {shots_path}")
    print()
    print("Next steps:")
    print(f"  - Review/edit {SHOTS_FILE} to remove false positives, then:")
    print(f"      python3 {sys.argv[0]} compile {video_dir}")


def _generate_sfx(path, sample_rate=44100):
    """Generate a punchy video-game-style shotgun sound effect as WAV."""
    import wave

    duration = 0.5
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)

    noise = np.random.default_rng(42).standard_normal(n)
    attack = noise * np.exp(-t * 60) * 0.7

    freq = 180 * np.exp(-t * 4) + 40
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    boom = np.sin(phase) * np.exp(-t * 5) * 1.0

    crack = noise * np.exp(-t * 35) * 0.5

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
    shots_path = os.path.join(video_dir, SHOTS_FILE)
    if not os.path.exists(shots_path):
        print(f"No {shots_path} found. Run detect first:")
        print(f"  python3 {sys.argv[0]} detect")
        sys.exit(1)

    with open(shots_path) as f:
        entries = json.load(f)

    if not entries:
        print("No entries in JSON file.")
        sys.exit(1)

    if merge:
        from collections import OrderedDict
        by_file = OrderedDict()
        for entry in entries:
            by_file.setdefault(entry["file"], []).append(entry)
        clips_to_extract = []
        for fname, group_entries in by_file.items():
            timestamps = sorted(e["timestamp"] for e in group_entries)
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
              f"({len(entries)} entries)...\n")
    else:
        clips_to_extract = [
            {"file": e["file"], "start_ts": e["timestamp"],
             "end_ts": e["timestamp"], "count": 1,
             "timestamps": [e["timestamp"]]}
            for e in entries
        ]
        print(f"Compiling {len(entries)} clips...\n")

    clip_dir = tempfile.mkdtemp(prefix="shots_")
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
            print(f"  [{i+1}/{total}] {clip['file']} @ {clip['start_ts']}s")

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

    shutil.rmtree(clip_dir)

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", output_path],
        capture_output=True, text=True
    )
    result_dur = float(json.loads(result.stdout)["format"]["duration"])
    print(f"\nDone! {output_path}")
    print(f"Duration: {result_dur:.1f}s ({len(clip_paths)} clips)")


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect gunshots in shotgun-camera videos and compile a highlight reel.",
        epilog="Examples:\n"
               "  python3 detect_shots.py detect ~/videos\n"
               "  python3 detect_shots.py compile ~/videos\n"
               "  python3 detect_shots.py ~/videos\n",
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
        help="merge multiple shots from the same file into one clip",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="burn filename and source timecode onto each clip",
    )
    parser.add_argument(
        "--sfx", action="store_true",
        help="replace gunshot audio with a video-game-style sound effect",
    )
    args = parser.parse_args()

    if args.command and args.command not in ("detect", "compile"):
        args.video_dir = args.command
        args.command = None

    check_dependencies()
    video_dir = os.path.abspath(args.video_dir)

    if args.command == "detect":
        cmd_detect(video_dir)
    elif args.command == "compile":
        cmd_compile(video_dir, merge=args.merge, overlay=args.overlay,
                    sfx=args.sfx)
    else:
        cmd_detect(video_dir)
        print()
        cmd_compile(video_dir, merge=args.merge, overlay=args.overlay,
                    sfx=args.sfx)


if __name__ == "__main__":
    main()
