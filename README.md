# Clay Target Hit Detector

Automatically detects clay target hits and shots fired in shotgun-mounted camera footage (e.g. ShotKam) and compiles them into a highlight reel.

## Requirements

- Python 3 with NumPy, Matplotlib, Pillow
- FFmpeg (provides `ffmpeg` and `ffprobe`)

```bash
brew install ffmpeg
pip install -r requirements.txt
```

## Usage

### Detect and compile in one step

```bash
python3 detect_hits.py /path/to/videos
```

Analyzes all `SHOT*.MP4` files in the directory, detects hits and shots fired, and produces `hits_compilation.mp4`.

### Two-step workflow (recommended)

**1. Detect**

```bash
python3 detect_hits.py detect /path/to/videos
```

Use `--type` to control what gets detected:

```bash
python3 detect_hits.py detect --type shot /path/to/videos   # audio gunshots only
python3 detect_hits.py detect --type hit /path/to/videos    # visual hits only
python3 detect_hits.py detect --type both /path/to/videos   # both (default)
```

Writes a `hits.json` file with detected entries:

```json
[
  { "file": "SHOT0001.MP4", "timestamp": 5.38, "type": "shot" },
  { "file": "SHOT0001.MP4", "timestamp": 8.96, "type": "hit" }
]
```

Each entry has a `type` field: `"shot"` for audio-detected gunshots, `"hit"` for visually-detected clay target breaks.

**2. Review and edit**

Open `hits.json` and remove any false positives or add missed entries manually.

**3. Compile video**

```bash
python3 detect_hits.py compile /path/to/videos
```

Reads `hits.json` and produces `hits_compilation.mp4` with a 1.5 second clip per entry (0.5s before, 1.0s after), with audio included.

Use `--merge` to combine multiple detections from the same file into one continuous clip (splits when the gap exceeds 5 seconds):

```bash
python3 detect_hits.py compile --merge /path/to/videos
```

Use `--overlay` to burn the source filename and timecode onto each clip for review:

```bash
python3 detect_hits.py compile --merge --overlay /path/to/videos
```

Use `--sfx` to replace gunshot audio with a synthetic video-game-style sound effect:

```bash
python3 detect_hits.py compile --merge --sfx /path/to/videos
```

All compile flags can be combined. The video directory defaults to the current directory if omitted.

## How detection works

### Visual hit detection

1. Each video is downscaled to 320px wide grayscale for fast analysis.
2. Frame-to-frame pixel differences are computed. A clay hit produces a sharp spike in this signal due to the sudden burst of fragments.
3. Spikes above 3 standard deviations from the mean are flagged as candidates.
4. Candidates where the frame brightness is well below average are rejected -- this filters out false positives from lowering the gun toward the ground after a shot.

### Audio shot detection

1. Raw PCM audio is extracted from each video via FFmpeg.
2. RMS energy is computed in 10ms non-overlapping windows.
3. Windows with energy above 5 standard deviations from the mean are identified as gunshot transients.
4. Candidates with peak RMS below 18000 are rejected -- this filters out the gun break-open sound, which is loud enough to spike the energy envelope but has lower sustained energy than an actual gunshot.
5. Spikes within 0.5 seconds are clustered, with the peak energy window selected as the shot timestamp.

## Limitations

- The visual hit detector cannot distinguish a hit from a close miss where the shot produces a visible smoke puff. Review `hits.json` to catch these.
- Audio shot detection requires the video to have an audio track. Videos without audio will only produce visual hit detections.
- Expects filenames matching `SHOT*.MP4`. Rename files or adjust the glob pattern in the script if needed.

## Shot Analysis Tool

Compares hit vs miss patterns to find what the shooter does differently when missing.

### Workflow

**Quick start (one command):**

```bash
python3 analyze_shots.py all /path/to/videos
```

This runs detection, generates `labels.json`, and produces the analysis report. It skips any step whose output already exists, so re-running after editing `labels.json` is safe.

**Step-by-step (recommended for accurate hit/miss labeling):**

**1. Detect shots and hits:**

```bash
python3 detect_hits.py detect --type both /path/to/videos
```

**2. Generate labels for review:**

```bash
python3 analyze_shots.py label /path/to/videos
```

This creates `labels.json` with one entry per shot, auto-classified as hit or miss.

**3. Review and correct labels:**

Open `labels.json` and change `"outcome"` to `"hit"` or `"miss"` for each shot. The auto-detection is a starting point -- correct any errors before analysis.

**4. Run analysis:**

```bash
python3 analyze_shots.py /path/to/videos
```

Uses `labels.json` if present, otherwise falls back to auto-detection.

### Output

- **Text report** (stdout): summary stats, signal comparison table with effect sizes, top distinguishing factors, per-miss breakdown
- **Plots** (saved to `analysis/`):
  - `diff_profiles.png` -- frame difference profiles aligned to shot (green=hits, red=misses)
  - `brightness_profiles.png` -- brightness trajectories showing gun tracking
  - `audio_profiles.png` -- audio energy profiles
  - `signal_distributions.png` -- boxplot comparison of all signals
  - `quadrant_motion.png` -- pre-shot camera motion direction scatter
  - `effect_sizes.png` -- ranked bar chart of which signals best separate hits from misses
- **Annotated miss frames** (saved to `analysis/`): frame at each miss moment with key stats compared to hit averages

### Signals analyzed

- **Pre-shot stability** -- frame-diff variance in 1s before shot (camera steadiness)
- **Pre-shot brightness slope** -- brightness trend before shot (gun tracking up/down)
- **Frame diff at shot** -- visual activity at the shot moment
- **Post-shot recovery** -- how quickly the frame stabilizes after the shot
- **Audio energy profile** -- peak RMS and decay rate of the gunshot sound
- **Quadrant motion** -- crude directional camera motion from per-quadrant brightness changes
