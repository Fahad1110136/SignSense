"""Per-user session logic of the SignSense app (no TensorFlow / Gradio imports, so it can be tested on its own).

A Session receives one webcam frame at a time, asks the recognizer what it sees, and feeds the result to the typing engine."""
import threading
import time
import numpy as np
import typing_engine as eng

STATIC, DYNAMIC = eng.STATIC_CLASSES, eng.DYNAMIC_CLASSES
UNIFORM = np.ones(len(STATIC)) / len(STATIC)
MIN_GESTURE_FRAMES = 12          # a recorded dynamic gesture needs at least this many frames with a visible hand


def top_k(probs, names, k=3):
    return [(names[i], float(probs[i])) for i in np.argsort(-np.asarray(probs))[:k]]


class Session:
    def __init__(self, recognizer, cfg, lexicon):
        self.rec, self.cfg = recognizer, cfg
        tcfg = eng.TyperConfig(**cfg['typer'])
        self.typer = eng.SignTyper(tcfg, lexicon)
        self.lock = threading.Lock()
        self.recording, self.buf = False, []
        self.cur_top3, self.last_top3 = [], []
        self.last_event, self.ms, self.pts = 'ready', 0.0, None

    # ---- settings coming from the UI sliders ----
    def apply_settings(self, letter_conf, stable_frames, correct_words):
        c = self.typer.cfg
        c.letter_conf, c.stable_frames, c.correct_words = float(letter_conf), int(stable_frames), bool(correct_words)

    # ---- what the UI shows ----
    def snapshot(self):
        t = self.typer
        hold = ''
        if t._cand is not None:
            need = t.cfg.stable_frames if t._cand[0] == 'letter' else t.cfg.control_frames
            name = t._cand[1] if t._cand[0] == 'letter' else t._cand[0]
            hold = f'holding **{name}**: {t._cand_n}/{need}'
        status = (f'**Last event:** {self.last_event}  \n'
                  + (f'**{hold}**  \n' if hold else '')
                  + (f'**Recording a dynamic gesture:** {len(self.buf)} frames  \n' if self.recording else '')
                  + f'**Speed:** {self.ms:.0f} ms per frame')
        tops = [f'{eng.LETTER_MAP[l]}  {l}  {p:.0%}' for l, p in self.last_top3]
        return {'text': t.text, 'status': status, 'top3': tops, 'pts': self.pts}

    # ---- manual controls (buttons) ----
    def manual(self, name):
        with self.lock:
            if name == 'clear':
                self.typer.reset(); self.last_top3 = []; self.last_event = 'cleared'
            elif name == 'backspace':
                a = self.typer._commit(('delete',)); self.last_event = f'manual backspace ({a["action"]})'
            elif name == 'space':
                a = self.typer._commit(('space',)); self.last_event = f'manual space ({a["action"]})'

    def replace_last(self, i):
        with self.lock:
            if self.typer.current and i < len(self.last_top3):
                label = self.last_top3[i][0]
                self.typer.current[-1] = eng.LETTER_MAP[label]; self.last_event = f'last letter replaced by {label}'

    # ---- dynamic letters: push-to-record ----
    def _finish_dynamic(self):
        if len(self.buf) < MIN_GESTURE_FRAMES:
            self.last_event = f'gesture too short or hand not visible ({len(self.buf)} frames)'; self.buf = []; return
        X, M = eng.resample_segment(self.buf, 88)
        probs = self.rec.dynamic_probs(eng.dynamic_features(X[None], M[None]))
        k = int(np.argmax(probs)); label, conf = DYNAMIC[k], float(probs[k])
        a = self.typer.step('dynamic', label, conf)
        self.last_top3 = top_k(probs, DYNAMIC, 3) if a else self.last_top3
        self.last_event = (f'dynamic letter {label} ({conf:.0%}) typed' if a else f'dynamic gesture {label} rejected (confidence {conf:.0%} is below {self.typer.cfg.dynamic_conf:.0%})')
        self.buf = []

    # ---- one frame ----
    def process(self, frame, record, mode, crop, mirror):
        if frame is None:
            return self.snapshot()
        if not self.lock.acquire(blocking=False):      # previous frame of this user still running: skip this one
            return self.snapshot()
        t0 = time.time()
        try:
            if mirror:
                frame = frame[:, ::-1]
            vec, self.pts = self.rec.landmarks(frame)
            if record:                                 # collecting a dynamic gesture
                if not self.recording:
                    self.recording, self.buf = True, []
                if vec is not None:
                    self.buf.append(vec)
                self.last_event = 'recording a dynamic gesture...'
            elif self.recording:                       # recording just ended: classify it
                self.recording = False; self._finish_dynamic()
            else:
                self._normal_frame(frame, vec, mode, crop)
        finally:
            self.ms = (time.time() - t0) * 1000
            self.lock.release()
        return self.snapshot()

    def _normal_frame(self, frame, vec, mode, crop):
        if vec is None:
            kind, label, conf = 'none', None, 0.0
            self.last_event = 'no hand (neutral)'
        else:
            ps, pdel = self.rec.control_probs(vec)
            kind, label, conf = eng.decide_event(True, ps, pdel, UNIFORM, STATIC, self.cfg['control']['threshold'])
            if kind == 'letter':
                probs = self.rec.letter_probs(frame, mode, crop, self.pts)
                self.cur_top3 = top_k(probs, STATIC, 3)
                label, conf = self.cur_top3[0]
                self.last_event = f'sees {label} ({conf:.0%})' + ('' if conf >= self.typer.cfg.letter_conf else ' - below the confidence setting')
            else:
                self.last_event = f'sees control gesture {kind} ({conf:.0%})'
        a = self.typer.step(kind, label, conf)
        if a:
            if a['action'] == 'letter':
                self.last_top3 = self.cur_top3
                self.last_event = f'typed {a["char"]} ({label})'
            elif a['action'] == 'space':
                self.last_event = 'space: word finished' + (f' and corrected to {a["word"]}' if a['corrected'] else '')
            elif a['action'] == 'delete':
                self.last_event = 'delete: removed the last character'
