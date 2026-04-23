"""Training driver: use videos/training/labels.json for per-shot ground
truth, run the per-clay-type classifier, report metrics grouped by
type and outcome to help find separating thresholds.

Filename convention for type: leading alphabetic prefix, one of
{away, coming, left, right} (e.g. away_1_1_1.MP4, right0025.MP4).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_hits import (  # noqa: E402
    extract_audio, compute_audio_energy, find_gunshots, AUDIO_WINDOW_MS,
    compute_block_frames, classify_shot_as_hit,
)
from clay_track import classify_shot, classify_outcome  # noqa: E402

VIDEO_DIR = "videos/training"
LABELS = json.load(open(os.path.join(VIDEO_DIR, "labels.json")))
TYPE_RE = re.compile(r"^(away|coming|left|right)", re.IGNORECASE)


def parse_type(filename):
    m = TYPE_RE.match(filename)
    return m.group(1).lower() if m else None


def detect_shots(path):
    samples = extract_audio(path)
    energy = compute_audio_energy(samples)
    raw = find_gunshots(energy, samples=samples, window_ms=AUDIO_WINDOW_MS)
    return [r[0] for r in raw]


def fmt_post(post):
    parts = []
    for p in post:
        if p["found"]:
            parts.append(
                f".{p['score']:>4.1f}/{p['dist']:>3.0f}/n{p['n_blobs']}")
        else:
            parts.append("x          ")
    return " ".join(parts)


def main():
    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.MP4")))
    if not files:
        print(f"No videos in {VIDEO_DIR}")
        return

    rows = []
    for path in files:
        fname = os.path.basename(path)
        ctype = parse_type(fname)
        labels = LABELS.get(fname)
        if ctype is None or labels is None:
            print(f"skip (no type/labels): {fname}")
            continue
        shots = detect_shots(path)
        if len(shots) != len(labels):
            print(f"!! {fname}: {len(shots)} shots detected but "
                  f"{len(labels)} labels  (shots={shots})")
            continue
        # Pre-compute block frames once per video (only used for `away`).
        block_frames = block_fps = None
        if ctype == "away":
            block_frames, block_fps = compute_block_frames(path)
        print(f"\n=== {fname}  type={ctype}  shots={len(shots)}")
        for i, (ts, lab) in enumerate(zip(shots, labels), 1):
            if ctype == "away":
                bres = classify_shot_as_hit(block_frames, block_fps, ts)
                pred = "hit" if (bres and bres.get("is_hit")) else "miss"
                ok = (pred == lab)
                print(f"  s{i} t={ts:.2f} truth={lab:<4} pred={pred:<4} "
                      f"{'OK' if ok else 'WRONG'}  "
                      f"score={bres.get('score', 0):.1f} "
                      f"spread={bres.get('spread', 0)}")
                rows.append({"file": fname, "type": ctype, "ts": ts,
                             "label": lab, "pred": pred,
                             "method": "block",
                             "score": bres.get("score") if bres else 0,
                             "spread": bres.get("spread") if bres else 0})
                continue

            r = classify_shot(path, ts, ctype)
            decision = classify_outcome(r, ctype)
            pred = "hit" if decision["is_hit"] else "miss"
            ok = (pred == lab)
            if not r["ok"]:
                print(f"  s{i} t={ts:.2f} truth={lab:<4} pred={pred:<4} "
                      f"{'OK' if ok else 'WRONG'}  ({r['reason']})")
                rows.append({"file": fname, "type": ctype, "ts": ts,
                             "label": lab, "pred": pred, "method": "track",
                             "ok": False})
                continue
            ly, lx = r["track_last"][:2]
            print(f"  s{i} t={ts:.2f} truth={lab:<4} pred={pred:<4} "
                  f"{'OK' if ok else 'WRONG'}  "
                  f"ratio={decision['score']:.2f}  "
                  f"tlen={r['n_track']:>2}  pre={r['pre_score']:>4.1f}  "
                  f"last=({ly:>3},{lx:>3})")
            rows.append({"file": fname, "type": ctype, "ts": ts,
                         "label": lab, "pred": pred, "method": "track",
                         "ok": True,
                         "ratio": decision["score"],
                         "n_track": r["n_track"],
                         "pre_score": r["pre_score"],
                         "post": r["post"]})

    # Confusion matrix per type.
    print("\n========== confusion matrix per type ==========")
    print(f"{'type':<8} {'truth':<5} {'pred=hit':>9} {'pred=miss':>10} {'acc':>6}")
    by_t = {}
    for r in rows:
        by_t.setdefault(r["type"], []).append(r)
    overall = [0, 0]
    for ctype in sorted(by_t):
        rs = by_t[ctype]
        for lab in ("hit", "miss"):
            sub = [r for r in rs if r["label"] == lab]
            if not sub:
                continue
            ph = sum(1 for r in sub if r["pred"] == "hit")
            pm = sum(1 for r in sub if r["pred"] == "miss")
            correct = ph if lab == "hit" else pm
            print(f"{ctype:<8} {lab:<5} {ph:>9} {pm:>10} "
                  f"{correct}/{len(sub)}")
            overall[0] += correct
            overall[1] += len(sub)
    print(f"\nOVERALL: {overall[0]}/{overall[1]} correct "
          f"({100.0 * overall[0] / max(1, overall[1]):.0f}%)")

    with open("/tmp/train_clay_rows.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
