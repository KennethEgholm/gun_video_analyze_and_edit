# TODO List
## This file is for tracking tasks and features that need to be implemented in the Video detection project.
## Do NOT implement anything from this file until it has been moved to the main specification file (spec.md) and approved by the project manager.

### Ideas

- **CNN hit/miss classifier for clays**
  - Extract a short window (~10 frames, t+0.1s to t+0.5s) around each detected shot.
  - Crop ROI: start with fixed sky region; upgrade to motion-based crop (frame diff → brightest blob) if needed.
  - Labeling CLI: play each clip, prompt `h`/`m`/`s`, save to `dataset/hit/` and `dataset/miss/`. Aim for 200-500 labels.
  - Model: fine-tune pretrained MobileNetV3 / ResNet18 from torchvision, 2-class head.
    - v1: single peak frame input (simple, forgiving on crop alignment).
    - v2: stack N frames as 3N input channels to capture the breaking-clay puff (better accuracy, needs consistent crops).
  - Train CPU-only, ~5-10 epochs with flip/brightness augmentation.
  - Add `--classify` flag to `detect_shots.py` to emit hit/miss alongside each shot timestamp.
  - Alternative: call a vision-language model (Gemini Flash, Claude vision) per shot — no training, higher per-call cost.

### Issues
