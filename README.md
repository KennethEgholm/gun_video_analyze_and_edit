# Clay Shot Highlight Reel

Detects gunshots in shotgun-mounted camera footage (e.g. ShotKam) and compiles them into a highlight reel.

## Requirements

- Python 3 with NumPy, Pillow
- FFmpeg (provides `ffmpeg` and `ffprobe`)

```bash
brew install ffmpeg
pip install -r requirements.txt
```

## Usage

### Detect and compile in one step

```bash
python3 detect_shots.py /path/to/videos
```

Analyzes all `SHOT*.MP4` files in the directory, detects gunshots, and produces `shots_compilation.mp4`.

### Two-step workflow (recommended)

**1. Detect**

```bash
python3 detect_shots.py detect /path/to/videos
```

Writes a `shots.json` file with one entry per detected shot:

```json
[
  { "file": "SHOT0001.MP4", "timestamp": 5.81 }
]
```

**2. Review and edit**

Open `shots.json` and remove any false positives or add missed entries manually.

**3. Compile video**

```bash
python3 detect_shots.py compile /path/to/videos
```

Reads `shots.json` and produces `shots_compilation.mp4` with a 1.5 second clip per entry (0.5s before, 1.0s after), with audio included.

Use `--merge` to combine multiple shots from the same file into one continuous clip (splits when the gap exceeds 5 seconds):

```bash
python3 detect_shots.py compile --merge /path/to/videos
```

Use `--overlay` to burn the source filename and timecode onto each clip for review:

```bash
python3 detect_shots.py compile --merge --overlay /path/to/videos
```

Use `--sfx` to replace gunshot audio with a synthetic video-game-style sound effect:

```bash
python3 detect_shots.py compile --merge --sfx /path/to/videos
```

All compile flags can be combined. The video directory defaults to the current directory if omitted.

## How detection works

1. Raw PCM audio is extracted from each video via FFmpeg.
2. RMS energy is computed in 10ms non-overlapping windows.
3. Windows with energy above 5 standard deviations from the mean are identified as gunshot transients.
4. Candidates with peak RMS below 18000 are rejected — this filters out the gun break-open sound, which is loud enough to spike the energy envelope but has lower sustained energy than an actual gunshot.
5. Spikes within 0.5 seconds are clustered, with the peak energy window selected as the shot timestamp.

## Limitations

- Audio shot detection requires the video to have an audio track. Videos without audio cannot be analyzed.
- Expects filenames matching `SHOT*.MP4` (case-insensitive). Rename files or adjust the glob pattern in the script if needed.
