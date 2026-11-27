"""The two small scikit-learn models of the app (control-gesture classifier and dynamic Random Forest)."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import typing_engine as eng

FEAT = [f'f{i}' for i in range(63)]


def _mirror(F):
    G = F.reshape(-1, 21, 3).copy(); G[..., 0] *= -1
    return G.reshape(-1, 63)


def train_control(data_dir, classifier='HistGradientBoosting'):
    """space / delete / other from hand landmarks (recorded gestures + dataset negatives, both with mirrored copies)."""
    ctrl = pd.read_csv(os.path.join(data_dir, 'control_gestures.csv'))
    pos = ctrl[ctrl['label'].isin(['space', 'delete'])]
    Xp, yp = pos[FEAT].values.astype(np.float32), pos['label'].values
    neg = np.load(os.path.join(data_dir, 'control_negatives.npz'))
    Xn = np.concatenate([neg['letters'], neg['dynamic_frames'], ctrl[ctrl['label'] == 'other'][FEAT].values.astype(np.float32)])
    X = np.concatenate([Xp, _mirror(Xp), Xn, _mirror(Xn)])
    y = np.concatenate([yp, yp, np.array(['other'] * (2 * len(Xn)))])
    if classifier == 'RandomForest':
        clf = RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=-1, random_state=0)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight='balanced', random_state=0)
    return clf.fit(X, y)


def train_dynamic(data_dir):
    """Random Forest on summary statistics of the 88-frame landmark sequence (all 393 videos)."""
    d = np.load(os.path.join(data_dir, 'dynamic_landmarks.npz'))
    y = d['y'] - len(eng.STATIC_CLASSES)
    return RandomForestClassifier(400, random_state=0, n_jobs=-1).fit(eng.dynamic_features(d['X'], d['mask']), y)


def load_or_train(path, trainer):
    try:
        return joblib.load(path)
    except Exception as e:                                   # wrong scikit-learn version, missing file, ...
        print(f'[SignSense] Could not load {os.path.basename(path)} ({type(e).__name__}); retraining it from app/data ...')
        model = trainer()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path, compress=3)
        print(f'[SignSense] Retrained and saved {os.path.basename(path)}.')
        return model