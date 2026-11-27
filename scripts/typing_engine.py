"""
SignSense typing engine: turns per-frame recognition events into Urdu text.
No TensorFlow dependency. Used by the pipeline notebook (simulation) and by the web app.
Pieces
    LETTER_MAP / CHAR_TO_CLASS  dataset class name <-> Urdu character
    Lexicon                     dictionary correction (edit distance) applied when a word ends
    TyperConfig / SignTyper     letter buffer, stability check, neutral requirement, space and delete gestures
    MotionRouter                splits a landmark stream into STILL (-> static model) and MOVING segments (-> dynamic model)
    resample_segment, dynamic_features   turn a moving segment into the feature vector of the dynamic Random Forest
    decide_event                fuses control-gesture and letter probabilities of one frame into one event
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np

# Dataset class name -> Urdu character. Derived from the class numbers in the dataset file names (01 ... 40),
# which follow the order of the Urdu alphabet.
LETTER_MAP = {
    'Alifmad': 'آ', 'Alif': 'ا', 'Bay': 'ب', 'Pay': 'پ', 'Tay': 'ت', 'Taay': 'ٹ', 'Say': 'ث', 'Jeem': 'ج',
    'Chay': 'چ', '1-Hay': 'ح', 'Khay': 'خ', 'Dal': 'د', 'Daal': 'ڈ', 'Zaal': 'ذ', 'Ray': 'ر', 'Aray': 'ڑ',
    'Zay': 'ز', 'Zaey': 'ژ', 'Seen': 'س', 'Sheen': 'ش', 'Suad': 'ص', 'Zuad': 'ض', 'Tuey': 'ط', 'Zuey': 'ظ',
    'Ain': 'ع', 'Ghain': 'غ', 'Fay': 'ف', 'Kaf': 'ق', 'Kiaf': 'ک', 'Gaaf': 'گ', 'Lam': 'ل', 'Meem': 'م',
    'Nuun': 'ن', 'Nuungh': 'ں', 'Wao': 'و', '2-Hay': 'ہ', 'Dochahay': 'ھ', 'Hamza': 'ء', 'Cyeh': 'ی', 'Byeh': 'ے'}
CHAR_TO_CLASS = {v: k for k, v in LETTER_MAP.items()}
ALPHABET = set(LETTER_MAP.values())
DYNAMIC_CLASSES = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']
STATIC_CLASSES = sorted(set(LETTER_MAP) - set(DYNAMIC_CLASSES))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Lexicon:
    """Word list for correction. Only words made of the 40 dataset letters are kept."""
    def __init__(self, words, min_len=3):
        self.words = {w.strip() for w in words if w.strip() and set(w.strip()) <= ALPHABET}
        self.by_len = defaultdict(list)
        for w in self.words:
            self.by_len[len(w)].append(w)
        self.min_len = min_len

    def correct(self, word, max_dist=1):
        """Return (word, changed). Only an unambiguous closest word within max_dist replaces the input."""
        if len(word) < self.min_len or word in self.words:
            return word, False
        cands = []
        for L in range(len(word) - max_dist, len(word) + max_dist + 1):
            for w in self.by_len.get(L, []):
                d = edit_distance(word, w)
                if d <= max_dist:
                    cands.append((d, w))
        if not cands:
            return word, False
        cands.sort()
        best = [w for d, w in cands if d == cands[0][0]]
        return (best[0], True) if len(best) == 1 else (word, False)


@dataclass
class TyperConfig:
    letter_conf: float = 0.9       # minimum confidence of a static letter frame
    stable_frames: int = 8         # consecutive frames of the same letter needed to commit it
    dynamic_conf: float = 0.6      # minimum confidence of a dynamic-letter prediction (one-shot event)
    control_conf: float = 0.9      # minimum probability of a space / delete frame
    control_frames: int = 8        # consecutive frames needed to commit space / delete
    neutral_frames: int = 3        # consecutive 'no hand' frames that count as the neutral position
    require_neutral: str = 'same'  # 'same': the same letter twice needs a neutral gap; 'all': every letter does
    correct_words: bool = True     # dictionary correction when a word ends
    max_dist: int = 1


class SignTyper:
    def __init__(self, cfg=None, lexicon=None):
        self.cfg = cfg or TyperConfig()
        self.lexicon = lexicon
        self.reset()

    def reset(self):
        self.words = []            # finished words as (raw, final)
        self.current = []          # characters of the word being typed
        self.trailing_space = False
        self._cand, self._cand_n, self._locked, self._none_n = None, 0, None, 0

    # ---- text ----
    @property
    def text(self):
        parts = [final for _, final in self.words]
        if self.current:
            parts.append(''.join(self.current))
        s = ' '.join(parts)
        return s + (' ' if self.trailing_space else '')

    def final_text(self):
        """Finish the word in progress (with correction) and return the whole text."""
        if self.current:
            self._finish_word()
        return ' '.join(f for _, f in self.words)

    # ---- internals ----
    def _finish_word(self):
        raw = ''.join(self.current); final = raw
        if self.cfg.correct_words and self.lexicon is not None:
            final, _ = self.lexicon.correct(raw, self.cfg.max_dist)
        self.words.append((raw, final)); self.current = []
        return raw, final

    def _reset_candidate(self):
        self._cand, self._cand_n = None, 0

    def _commit(self, key):
        kind = key[0]
        if kind == 'letter':
            ch = LETTER_MAP[key[1]]; self.current.append(ch); self.trailing_space = False
            return {'action': 'letter', 'char': ch}
        if kind == 'space':
            if not self.current:
                return {'action': 'space_ignored'}
            raw, final = self._finish_word(); self.trailing_space = True
            return {'action': 'space', 'raw': raw, 'word': final, 'corrected': raw != final}
        if kind == 'delete':
            if self.current:
                return {'action': 'delete', 'char': self.current.pop()}
            if self.trailing_space and self.words:      # deleting the space re-opens the last word (as typed)
                raw, _ = self.words.pop(); self.current = list(raw); self.trailing_space = False
                return {'action': 'delete', 'char': ' '}
            return {'action': 'delete_ignored'}

    def _step_key(self, key, need):
        locked = self._locked
        if locked == key or (locked == 'ANY' and key[0] == 'letter'):
            self._reset_candidate(); return None
        if key == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = key, 1
        if self._cand_n >= need:
            self._reset_candidate()
            self._locked = 'ANY' if (key[0] == 'letter' and self.cfg.require_neutral == 'all') else key
            return self._commit(key)
        return None

    # ---- main entry point: call once per frame ----
    def step(self, kind, label=None, conf=0.0):
        """kind: 'none' (no hand) | 'letter' (static letter) | 'dynamic' (one-shot dynamic letter) | 'space' | 'delete'."""
        c = self.cfg
        if kind == 'none':
            self._none_n += 1; self._reset_candidate()
            if self._none_n >= c.neutral_frames:
                self._locked = None
            return None
        self._none_n = 0
        if kind == 'dynamic':
            self._reset_candidate()
            return self._commit(('letter', label)) if conf >= c.dynamic_conf else None
        if kind == 'letter':
            if conf < c.letter_conf:
                self._reset_candidate(); return None
            return self._step_key(('letter', label), c.stable_frames)
        if kind in ('space', 'delete'):
            if conf < c.control_conf:
                self._reset_candidate(); return None
            return self._step_key((kind,), c.control_frames)
        raise ValueError(kind)


def decide_event(hand_detected, p_space, p_delete, letter_probs, letter_names, control_conf=0.9):
    """One frame -> (kind, label, confidence). Control gestures take priority over letters."""
    if not hand_detected:
        return 'none', None, 0.0
    if p_space >= control_conf and p_space >= p_delete:
        return 'space', None, float(p_space)
    if p_delete >= control_conf:
        return 'delete', None, float(p_delete)
    i = int(np.argmax(letter_probs))
    return 'letter', letter_names[i], float(letter_probs[i])


def resample_segment(seg, T=88):
    """List/array of (63,) landmark vectors -> (X (T,63) float32, M (T,) bool), uniformly resampled like Phase 1."""
    seg = np.asarray(seg, np.float32)
    idx = np.linspace(0, len(seg) - 1, T).round().astype(int)
    return seg[idx], np.ones(T, bool)


def dynamic_features(X, M):
    """Same features as the dynamic Random Forest of Phase 2. X: (N,T,63), M: (N,T) bool -> (N, 508)."""
    X = np.asarray(X, np.float32); M = np.asarray(M, bool)
    F = np.concatenate([X, M[..., None].astype('float32')], axis=-1)
    V = np.zeros_like(X); V[:, 1:] = (X[:, 1:] - X[:, :-1]) * (M[:, 1:] & M[:, :-1])[..., None]
    F = np.concatenate([F, V], axis=-1).astype('float32')
    return np.concatenate([F.mean(1), F.std(1), F.min(1), F.max(1)], axis=1)


def window_extent(W):
    """Motion measure of a window of landmark vectors: largest RMS distance of any frame from the window mean."""
    W = np.asarray(W, np.float32)
    return float(np.sqrt(((W - W.mean(0)) ** 2).mean(1)).max())


class MotionRouter:
    """Streaming splitter. update(vec) -> (state, segment). vec = 63 normalized landmarks or None (no hand).
    state is 'none', 'still' or 'moving'; segment is an (n,63) array when a moving segment just ended."""
    def __init__(self, tau, window=10, still_frames=6, min_segment=12, max_segment=150):
        self.tau, self.window, self.still_frames = tau, window, still_frames
        self.min_segment, self.max_segment = min_segment, max_segment
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.window); self.moving = False; self.seg = []; self.still_n = 0

    def _close(self):
        seg = self.seg[:len(self.seg) - self.still_n] if self.still_n else self.seg
        self.moving, self.seg, self.still_n = False, [], 0
        return np.asarray(seg, np.float32) if len(seg) >= self.min_segment else None

    def update(self, vec):
        if vec is None:
            out = self._close() if self.moving else None
            self.buf.clear()
            return 'none', out
        self.buf.append(np.asarray(vec, np.float32))
        if len(self.buf) < self.window:
            return ('moving' if self.moving else 'still'), None
        e = window_extent(self.buf)
        if not self.moving:
            if e > self.tau:
                self.moving = True; self.seg = list(self.buf); self.still_n = 0
                return 'moving', None
            return 'still', None
        self.seg.append(np.asarray(vec, np.float32))
        self.still_n = self.still_n + 1 if e <= self.tau else 0
        if self.still_n >= self.still_frames or len(self.seg) >= self.max_segment:
            return 'still', self._close()
        return 'moving', None