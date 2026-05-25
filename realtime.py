#!/usr/bin/env python3
"""
realtime.py — live fire/smoke detection from webcam or video file.

Usage:
 python realtime.py --source 0 # default webcam
 python realtime.py --source path/to/video.mp4
 python realtime.py --source 0 --model outputs/models/mobilenetv2_best.keras

Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ----- Config (must match the notebook) -----
IMG_SIZE = 224
CLASSES = ["non_fire", "fire", "smoke"]


# ----- Classical pre-filter (same rules as the notebook) -----
def flame_mask_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([0, 100, 100]),
        np.array([35, 255, 255])
    )


def flame_mask_rgb(img_bgr):
    b, g, r = cv2.split(img_bgr)

    r_mean = r.mean()

    rule1 = (r > g) & (g > b)
    rule2 = r > r_mean

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]

    S_T = 60

    rule3 = s >= (
        (255 - r.astype(np.int32)) * S_T // max(int(r_mean), 1)
    )

    return ((rule1 & rule2 & rule3).astype(np.uint8) * 255)


def morphological_clean(mask, k=5):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)

    return mask


def classical_pre_filter(img_bgr, fire_threshold=0.005):
    """
    Returns True if frame is potentially fire/smoke
    (so we should run CNN).
    """

    flame = cv2.bitwise_or(
        flame_mask_hsv(img_bgr),
        flame_mask_rgb(img_bgr)
    )

    flame = morphological_clean(flame)

    fire_ratio = (
        (flame > 0).sum()
        / (img_bgr.shape[0] * img_bgr.shape[1])
    )

    return fire_ratio > fire_threshold, fire_ratio


# ----- CNN preprocessing (same as the notebook) -----
def preprocess(img_bgr):
    img = cv2.resize(
        img_bgr,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    img = cv2.GaussianBlur(img, (3, 3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    img = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    return img


def main():
    ap = argparse.ArgumentParser(
        description="Real-time fire/smoke detection"
    )

    ap.add_argument(
        "--source",
        default=0,
        help="Webcam index (e.g. 0) or path to video file"
    )

    ap.add_argument(
        "--model",
        default="outputs/models/mobilenetv2_best.keras",
        help="Path to the saved Keras model"
    )

    ap.add_argument(
        "--every-n",
        type=int,
        default=2,
        help="Run the CNN every N flagged frames (default 2)"
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Fire-pixel ratio threshold for the pre-filter"
    )

    ap.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable the classical pre-filter (always run CNN)"
    )

    args = ap.parse_args()

    src = (
        int(args.source)
        if str(args.source).isdigit()
        else args.source
    )

    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train the model first via the notebook, or pass --model.")
        return

    print(f"[INFO] Loading model from {model_path} ...")

    model = tf.keras.models.load_model(str(model_path))

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {src}")
        return

    print("[INFO] Running. Press 'q' to quit.")

    last_label, last_prob = "non_fire", 0.0

    cnn_calls = 0
    skipped = 0
    frame_idx = 0

    fps_t0 = time.time()
    fps_frames = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_idx += 1
        fps_frames += 1

        # Pre-filter
        if args.no_prefilter:
            should_run = True
            ratio = 1.0
        else:
            should_run, ratio = classical_pre_filter(
                frame,
                args.threshold
            )

        # CNN (sub-sampled)
        if should_run and (frame_idx % args.every_n == 0):
            x = preprocess(frame)

            p = model.predict(
                x[None, ...],
                verbose=0
            )[0]

            last_label = CLASSES[int(p.argmax())]
            last_prob = float(p.max())

            cnn_calls += 1

        elif not should_run:
            last_label = "non_fire"
            last_prob = 1 - ratio

            skipped += 1

        # Display
        color = {
            "fire": (0, 0, 255),
            "smoke": (0, 165, 255),
            "non_fire": (0, 255, 0)
        }[last_label]

        cv2.putText(
            frame,
            f"{last_label.upper()} {last_prob:.2f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        cv2.putText(
            frame,
            f"flame_ratio={ratio:.3f} CNN calls={cnn_calls} skipped={skipped}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # FPS once per second
        if time.time() - fps_t0 >= 1.0:
            fps = fps_frames / (time.time() - fps_t0)

            print(
                f"[FPS] {fps:.1f} "
                f"cnn_calls={cnn_calls} "
                f"skipped={skipped} "
                f"last={last_label}"
            )

            fps_t0 = time.time()
            fps_frames = 0

        cv2.imshow(
            "Fire/Smoke Detection — q to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(
        f"[DONE] frames={frame_idx} "
        f"cnn_calls={cnn_calls} "
        f"skipped={skipped}"
    )


if __name__ == "__main__":
    main()