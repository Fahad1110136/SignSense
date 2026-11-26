"""
Record hand-landmark samples for the control gestures of SignSense.

Labels (press the key to START recording, press it again to STOP):
    1 = space   (gesture that ends a word)
    2 = delete  (gesture that removes the last letter)
    3 = other   (everything that must NOT trigger space/delete: relaxed hand, hand in
                 transition, and your own attempts at ordinary Urdu letter signs)
    q = quit

Only normalized hand landmarks are stored (no images, no faces). The normalization is
identical to the one used for the dynamic letters in Phase 1: wrist moved to the origin,
all points divided by the largest wrist-to-landmark distance.

Usage:
    python scripts/collect_gestures.py --person fahad
Output (appended): dataset/control_gestures.csv
"""

import argparse, csv, os, time, urllib.request
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
LABELS = {ord("1"): "space", ord("2"): "delete", ord("3"): "other"}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
SAMPLE_EVERY = 0.1          # Seconds between stored samples (about 10 per second)

def normalize(lm, w, h):
    a = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], np.float32)
    a -= a[0]
    sc = np.linalg.norm(a[:, :2], axis=1).max()
    return (a / sc).ravel() if sc > 1e-6 else None

def draw_hand(img, lm):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="who is recording (e.g. aamina, wisha, fahad)")
    ap.add_argument("--out", default="dataset/control_gestures.csv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=40, help="auto-stop each recording after this time")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.5))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    wr = csv.writer(f)
    if new_file:
        wr.writerow(["person", "session", "label", "hand"] + [f"f{i}" for i in range(63)])

    session = time.strftime("%Y%m%d_%H%M%S")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera (try --camera 1)")
    counts = {v: 0 for v in LABELS.values()}
    state, t_start, t_last = None, 0.0, 0.0
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        s = 480 / max(frame.shape[:2])                  
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        h, w = small.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hand = res.hand_landmarks[0] if res.hand_landmarks else None

        now = time.time()
        if state and now - t_start > args.max_seconds:
            state = None
        if state and hand is not None and now - t_last >= SAMPLE_EVERY:
            vec = normalize(hand, w, h)
            if vec is not None:
                side = res.handedness[0][0].category_name if res.handedness else ""
                wr.writerow([args.person, session, state, side] + [f"{v:.5f}" for v in vec]); f.flush()
                counts[state] += 1; t_last = now

        if hand is not None:
            draw_hand(small, hand)
        view = cv2.flip(small, 1)                           # mirror only the preview, detection used the raw frame
        status = f"RECORDING: {state}" if state else "ready"
        cv2.putText(view, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if state else (0, 200, 0), 2)
        cv2.putText(view, "hand: " + ("yes" if hand is not None else "NO"), (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, "  ".join(f"{k}:{v}" for k, v in counts.items()), (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(view, "1=space 2=delete 3=other q=quit", (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("SignSense gesture collection", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in LABELS:
            new = LABELS[key]
            state = None if state == new else new
            t_start = now

    cap.release(); cv2.destroyAllWindows(); f.close()
    print("Saved this session:", counts, "->", args.out)

if __name__ == "__main__":
    main()