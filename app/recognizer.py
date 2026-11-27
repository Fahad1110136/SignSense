"""Models of the SignSense app: MediaPipe hand landmarks, control-gesture classifier, static letter models, dynamic Random Forest."""
import os
import threading
import urllib.request
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import keras
import typing_engine as eng
from light_models import load_or_train, train_control, train_dynamic

STATIC = eng.STATIC_CLASSES
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Recognizer:
    def __init__(self, base_dir, cfg):
        self.cfg = cfg
        mdir = os.path.join(base_dir, 'models')
        lm_path = os.path.join(base_dir, 'hand_landmarker.task')
        if not os.path.exists(lm_path):
            urllib.request.urlretrieve(cfg['landmarker_model_url'], lm_path)
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=lm_path), running_mode=vision.RunningMode.IMAGE,
            num_hands=1, min_hand_detection_confidence=0.5))
        self.lm_lock = threading.Lock()                       # MediaPipe is not thread-safe
        ddir = os.path.join(base_dir, 'data')
        self.ctrl = load_or_train(os.path.join(mdir, cfg['control']['file']), lambda: train_control(ddir, cfg['control'].get('classifier', 'HistGradientBoosting')))
        assert list(self.ctrl.classes_) == cfg['control']['classes']
        self.i_space, self.i_delete = cfg['control']['classes'].index('space'), cfg['control']['classes'].index('delete')
        self.rf = load_or_train(os.path.join(mdir, cfg['dynamic']['file']), lambda: train_dynamic(ddir))
        self.fill = np.array(cfg['static']['pad_fill_rgb'], np.uint8)
        self.models = {f: keras.models.load_model(os.path.join(mdir, f), compile=False) for f in cfg['static']['files']}
        self.mobilenet = next(m for f, m in self.models.items() if 'mobilenetv2' in f)
        self.has_ensemble = len(self.models) == 3
        dummy = np.zeros((1, 224, 224, 3), np.float32)
        for m in self.models.values():                        # warm-up so the first live frame is not slow
            m(dummy, training=False)

    # ---- hand landmarks (same normalization as Phase 1) ----
    def landmarks(self, rgb):
        h, w = rgb.shape[:2]
        s = 480 / max(h, w)
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
        sh, sw = small.shape[:2]
        with self.lm_lock:
            res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
        if not res.hand_landmarks:
            return None, None
        lm = res.hand_landmarks[0]
        pts = np.array([[p.x, p.y] for p in lm], np.float32)      # normalized 0..1, used for drawing and cropping
        a = np.array([[p.x * sw, p.y * sh, p.z * sw] for p in lm], np.float32)
        a -= a[0]
        sc = np.linalg.norm(a[:, :2], axis=1).max()
        return ((a / sc).ravel() if sc > 1e-6 else None), pts

    def control_probs(self, vec):
        p = self.ctrl.predict_proba(vec[None])[0]
        return float(p[self.i_space]), float(p[self.i_delete])

    # ---- static letters ----
    def _prepare(self, rgb, crop, pts):
        img = rgb
        if crop and pts is not None:
            h, w = rgb.shape[:2]
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.8
            xa, xb = int(max(0, (cx - side / 2) * w)), int(min(w, (cx + side / 2) * w))
            ya, yb = int(max(0, (cy - side / 2) * h)), int(min(h, (cy + side / 2) * h))
            if xb - xa > 20 and yb - ya > 20:
                img = rgb[ya:yb, xa:xb]
        h, w = img.shape[:2]
        s = 224 / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        canvas = np.empty((224, 224, 3), np.uint8); canvas[:] = self.fill
        top, left = (224 - nh) // 2, (224 - nw) // 2
        canvas[top:top + nh, left:left + nw] = r
        return canvas[None].astype(np.float32)

    def letter_probs(self, rgb, mode, crop, pts):
        x = self._prepare(rgb, crop, pts)
        models = list(self.models.values()) if (mode == 'accuracy' and self.has_ensemble) else [self.mobilenet]
        return np.mean([m(x, training=False).numpy()[0] for m in models], axis=0)

    # ---- dynamic letters ----
    def dynamic_probs(self, feats):
        return self.rf.predict_proba(feats)[0]