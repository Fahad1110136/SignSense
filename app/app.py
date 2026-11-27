"""SignSense: Urdu Sign Language fingerspelling to text. Run with:  python app.py"""
import inspect
import json
import os
import cv2
import numpy as np
import gradio as gr
import typing_engine as eng
from recognizer import Recognizer, CONNECTIONS, STATIC
from session import Session, DYNAMIC, MIN_GESTURE_FRAMES, top_k

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(BASE, 'app_config.json'), encoding='utf-8'))
LEX = eng.Lexicon(open(os.path.join(BASE, CFG['lexicon_file']), encoding='utf-8').read().split())
REC = Recognizer(BASE, CFG)

ACC, FAST = 'Accuracy (3 models, slower)', 'Fast (MobileNetV2 only)'
MODES = [ACC, FAST] if REC.has_ensemble else [FAST]
DEFAULT_MODE = ACC if (CFG['static']['mode'] == 'ensemble' and REC.has_ensemble) else FAST
DEFAULT_STABLE = CFG['typer']['stable_frames'] if DEFAULT_MODE == FAST else 3
mode_key = lambda m: 'accuracy' if m == ACC else 'fast'

CSS = """
.urdu textarea { font-size: 34px !important; line-height: 2.1 !important; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Noto Naskh Arabic', serif !important; }
.sugg button { font-size: 22px !important; }
"""
HEAD = '<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">'
CHAR = lambda label: eng.LETTER_MAP[label]


def draw(frame, pts, status):
    img = np.ascontiguousarray(frame).copy()
    h, w = img.shape[:2]
    if pts is not None:
        p = [(int(x * w), int(y * h)) for x, y in pts]
        for a, b in CONNECTIONS:
            cv2.line(img, p[a], p[b], (0, 255, 0), 2)
        for q in p:
            cv2.circle(img, q, 3, (255, 0, 0), -1)
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return img


def sugg(r, i):
    return gr.update(value=r['top3'][i], visible=True) if i < len(r['top3']) else gr.update(value='-', visible=True)


def on_frame(frame, sess, record, mode, conf, stable, correct, crop, mirror):
    if frame is None:
        r = sess.snapshot(); return None, r['text'], r['status'], sugg(r, 0), sugg(r, 1), sugg(r, 2), sess
    sess.apply_settings(conf, stable, correct)
    r = sess.process(frame, record, mode_key(mode), crop, mirror)
    short = sess.last_event if len(sess.last_event) < 60 else sess.last_event[:57] + '...'
    return draw(frame[:, ::-1] if mirror else frame, r['pts'], short), r['text'], r['status'], sugg(r, 0), sugg(r, 1), sugg(r, 2), sess


# ---------- upload tabs (no webcam needed) ----------
def predict_photo(img, mode, crop):
    if img is None:
        return {}
    vec, pts = REC.landmarks(img)
    probs = REC.letter_probs(img, mode_key(mode), crop, pts)
    return {f'{CHAR(l)}  {l}': p for l, p in top_k(probs, STATIC, 5)}


def predict_video(path):
    if path is None:
        return {}, 'Upload a short video of one dynamic letter.'
    cap = cv2.VideoCapture(path); vecs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        v, _ = REC.landmarks(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if v is not None:
            vecs.append(v)
    cap.release()
    if len(vecs) < MIN_GESTURE_FRAMES:
        return {}, f'The hand was visible in only {len(vecs)} frames; at least {MIN_GESTURE_FRAMES} are needed.'
    X, M = eng.resample_segment(vecs, 88)
    probs = REC.dynamic_probs(eng.dynamic_features(X[None], M[None]))
    return {f'{CHAR(l)}  {l}': p for l, p in top_k(probs, DYNAMIC, 4)}, f'Used {len(vecs)} frames with a visible hand.'


# ---------- Gradio version compatibility ----------
# Gradio 6 removed Image(mirror_webcam=...) (now webcam_options=WebcamOptions(mirror=...))
# and moved Blocks(css=, head=) to launch(css=, head=). Detect which API is installed.
_img_params = inspect.signature(gr.Image.__init__).parameters
CAM_KW = {}
if 'mirror_webcam' in _img_params:
    CAM_KW['mirror_webcam'] = False
elif 'webcam_options' in _img_params and hasattr(gr, 'WebcamOptions'):
    CAM_KW['webcam_options'] = gr.WebcamOptions(mirror=False)

_blocks_takes_css = 'css' in inspect.signature(gr.Blocks.__init__).parameters
BLOCKS_KW = dict(css=CSS, head=HEAD) if _blocks_takes_css else {}
LAUNCH_KW = {} if _blocks_takes_css else dict(css=CSS, head=HEAD)


# ---------- interface ----------
with gr.Blocks(title='SignSense', **BLOCKS_KW) as demo:
    gr.Markdown('# SignSense: Urdu Sign Language to Text\nSpell a word letter by letter in Urdu Sign Language and watch it appear as Urdu text.')
    with gr.Tabs():
        with gr.Tab('Live typing (webcam)'):
            state = gr.State(lambda: Session(REC, CFG, LEX))
            with gr.Row():
                with gr.Column(scale=1):
                    cam = gr.Image(sources=['webcam'], type='numpy', label='Your camera', **CAM_KW)
                    record = gr.Checkbox(label='Record a dynamic letter (آ ڑ ج ہ): tick, perform the motion, then untick', value=False)
                    with gr.Accordion('Settings', open=False):
                        mode = gr.Radio(MODES, value=DEFAULT_MODE, label='Recognition mode')
                        conf = gr.Slider(0.3, 0.95, value=CFG['typer']['letter_conf'], step=0.05, label='Minimum confidence of a letter')
                        stable = gr.Slider(2, 10, value=DEFAULT_STABLE, step=1, label='Frames a letter must be held (stability)')
                        correct = gr.Checkbox(value=True, label='Dictionary correction when a word ends')
                        crop = gr.Checkbox(value=True, label='Crop to the hand before recognizing the letter')
                        mirror = gr.Checkbox(value=False, label='Mirror the camera image (try this if letters are not recognized)')
                with gr.Column(scale=1):
                    out_img = gr.Image(label='What the system sees', interactive=False)
                    text = gr.Textbox(label='Typed text', rtl=True, lines=2, interactive=False, elem_classes='urdu')
                    status = gr.Markdown()
                    gr.Markdown('**Not the right letter? Replace the last letter with one of the next best guesses:**')
                    with gr.Row(elem_classes='sugg'):
                        s1, s2, s3 = gr.Button('-'), gr.Button('-'), gr.Button('-')
                    with gr.Row():
                        b_back, b_space, b_clear = gr.Button('Backspace'), gr.Button('Add space'), gr.Button('Clear all', variant='stop')
            cam.stream(on_frame, inputs=[cam, state, record, mode, conf, stable, correct, crop, mirror],
                       outputs=[out_img, text, status, s1, s2, s3, state], time_limit=180, stream_every=0.2, concurrency_limit=4)
            for i, b in enumerate([s1, s2, s3]):
                b.click(lambda s, i=i: (s.replace_last(i), s.snapshot()['text'], s)[1:], [state], [text, state])
            b_back.click(lambda s: (s.manual('backspace'), s.snapshot()['text'], s)[1:], [state], [text, state])
            b_space.click(lambda s: (s.manual('space'), s.snapshot()['text'], s)[1:], [state], [text, state])
            b_clear.click(lambda s: (s.manual('clear'), s.snapshot()['text'], s)[1:], [state], [text, state])
        with gr.Tab('Letter from a photo'):
            gr.Markdown('Upload a photo of one **static** letter sign (works without a webcam).')
            with gr.Row():
                photo = gr.Image(type='numpy', sources=['upload'], label='Photo')
                with gr.Column():
                    pmode = gr.Radio(MODES, value=DEFAULT_MODE, label='Recognition mode')
                    pcrop = gr.Checkbox(value=False, label='Crop to the hand first')
                    plabel = gr.Label(num_top_classes=5, label='Most likely letters')
            photo.change(predict_photo, [photo, pmode, pcrop], plabel)
        with gr.Tab('Dynamic letter from a video'):
            gr.Markdown('Upload a short video of one **dynamic** letter (آ ڑ ج ہ).')
            with gr.Row():
                vid = gr.Video(label='Video')
                with gr.Column():
                    vlabel = gr.Label(num_top_classes=4, label='Most likely letters'); vinfo = gr.Markdown()
            vid.change(predict_video, [vid], [vlabel, vinfo])
        with gr.Tab('How to use'):
            gr.Markdown(f"""
**Control gestures** (hold for about a second): **space** = open flat palm (ends the word and applies dictionary correction), **delete** = thumbs down (removes the last character).

**Static letters (36):** hold the sign steady in front of the camera until the hold counter reaches its target. To type the same letter twice, drop your hand out of view for a moment in between.

**Dynamic letters (آ ڑ ج ہ):** tick the *Record* box, perform the motion once, untick the box.

**Tips:** good light, plain background, hand fully visible. If recognition is poor, try the *Mirror* setting and the *Crop to the hand* setting.

**Known limits:** about 87-90% of single letters are recognized correctly for new people, so names are often not typed perfectly; use the suggestion buttons to fix a letter. The dictionary only knows about {len(LEX.words)} demo words.
""")
            gp = os.path.join(BASE, 'assets', 'control_gestures.jpg')
            if os.path.exists(gp):
                gr.Image(gp, label='Control gestures: open palm = space, thumbs down = delete', interactive=False)

if __name__ == '__main__':
    demo.queue().launch(**LAUNCH_KW)










# """SignSense: Urdu Sign Language fingerspelling to text. Run with:  python app.py"""
# import json
# import os
# import cv2
# import numpy as np
# import gradio as gr
# import typing_engine as eng
# from recognizer import Recognizer, CONNECTIONS, STATIC
# from session import Session, DYNAMIC, MIN_GESTURE_FRAMES, top_k

# BASE = os.path.dirname(os.path.abspath(__file__))
# CFG = json.load(open(os.path.join(BASE, 'app_config.json'), encoding='utf-8'))
# LEX = eng.Lexicon(open(os.path.join(BASE, CFG['lexicon_file']), encoding='utf-8').read().split())
# REC = Recognizer(BASE, CFG)

# ACC, FAST = 'Accuracy (3 models, slower)', 'Fast (MobileNetV2 only)'
# MODES = [ACC, FAST] if REC.has_ensemble else [FAST]
# DEFAULT_MODE = ACC if (CFG['static']['mode'] == 'ensemble' and REC.has_ensemble) else FAST
# DEFAULT_STABLE = CFG['typer']['stable_frames'] if DEFAULT_MODE == FAST else 3
# mode_key = lambda m: 'accuracy' if m == ACC else 'fast'

# CSS = """
# .urdu textarea { font-size: 34px !important; line-height: 2.1 !important; font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Noto Naskh Arabic', serif !important; }
# .sugg button { font-size: 22px !important; }
# """
# HEAD = '<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">'
# CHAR = lambda label: eng.LETTER_MAP[label]


# def draw(frame, pts, status):
#     img = np.ascontiguousarray(frame).copy()
#     h, w = img.shape[:2]
#     if pts is not None:
#         p = [(int(x * w), int(y * h)) for x, y in pts]
#         for a, b in CONNECTIONS:
#             cv2.line(img, p[a], p[b], (0, 255, 0), 2)
#         for q in p:
#             cv2.circle(img, q, 3, (255, 0, 0), -1)
#     cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#     return img


# def sugg(r, i):
#     return gr.update(value=r['top3'][i], visible=True) if i < len(r['top3']) else gr.update(value='-', visible=True)


# def on_frame(frame, sess, record, mode, conf, stable, correct, crop, mirror):
#     if frame is None:
#         r = sess.snapshot(); return None, r['text'], r['status'], sugg(r, 0), sugg(r, 1), sugg(r, 2), sess
#     sess.apply_settings(conf, stable, correct)
#     r = sess.process(frame, record, mode_key(mode), crop, mirror)
#     short = sess.last_event if len(sess.last_event) < 60 else sess.last_event[:57] + '...'
#     return draw(frame[:, ::-1] if mirror else frame, r['pts'], short), r['text'], r['status'], sugg(r, 0), sugg(r, 1), sugg(r, 2), sess


# # ---------- upload tabs (no webcam needed) ----------
# def predict_photo(img, mode, crop):
#     if img is None:
#         return {}
#     vec, pts = REC.landmarks(img)
#     probs = REC.letter_probs(img, mode_key(mode), crop, pts)
#     return {f'{CHAR(l)}  {l}': p for l, p in top_k(probs, STATIC, 5)}


# def predict_video(path):
#     if path is None:
#         return {}, 'Upload a short video of one dynamic letter.'
#     cap = cv2.VideoCapture(path); vecs = []
#     while True:
#         ok, fr = cap.read()
#         if not ok:
#             break
#         v, _ = REC.landmarks(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
#         if v is not None:
#             vecs.append(v)
#     cap.release()
#     if len(vecs) < MIN_GESTURE_FRAMES:
#         return {}, f'The hand was visible in only {len(vecs)} frames; at least {MIN_GESTURE_FRAMES} are needed.'
#     X, M = eng.resample_segment(vecs, 88)
#     probs = REC.dynamic_probs(eng.dynamic_features(X[None], M[None]))
#     return {f'{CHAR(l)}  {l}': p for l, p in top_k(probs, DYNAMIC, 4)}, f'Used {len(vecs)} frames with a visible hand.'


# # ---------- interface ----------
# with gr.Blocks(title='SignSense', css=CSS, head=HEAD) as demo:
#     gr.Markdown('# SignSense: Urdu Sign Language to Text\nSpell a word letter by letter in Urdu Sign Language and watch it appear as Urdu text.')
#     with gr.Tabs():
#         with gr.Tab('Live typing (webcam)'):
#             state = gr.State(lambda: Session(REC, CFG, LEX))
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     cam = gr.Image(sources=['webcam'], type='numpy', label='Your camera', mirror_webcam=False)
#                     record = gr.Checkbox(label='Record a dynamic letter (آ ڑ ج ہ): tick, perform the motion, then untick', value=False)
#                     with gr.Accordion('Settings', open=False):
#                         mode = gr.Radio(MODES, value=DEFAULT_MODE, label='Recognition mode')
#                         conf = gr.Slider(0.3, 0.95, value=CFG['typer']['letter_conf'], step=0.05, label='Minimum confidence of a letter')
#                         stable = gr.Slider(2, 10, value=DEFAULT_STABLE, step=1, label='Frames a letter must be held (stability)')
#                         correct = gr.Checkbox(value=True, label='Dictionary correction when a word ends')
#                         crop = gr.Checkbox(value=True, label='Crop to the hand before recognizing the letter')
#                         mirror = gr.Checkbox(value=False, label='Mirror the camera image (try this if letters are not recognized)')
#                 with gr.Column(scale=1):
#                     out_img = gr.Image(label='What the system sees', interactive=False)
#                     text = gr.Textbox(label='Typed text', rtl=True, lines=2, interactive=False, elem_classes='urdu')
#                     status = gr.Markdown()
#                     gr.Markdown('**Not the right letter? Replace the last letter with one of the next best guesses:**')
#                     with gr.Row(elem_classes='sugg'):
#                         s1, s2, s3 = gr.Button('-'), gr.Button('-'), gr.Button('-')
#                     with gr.Row():
#                         b_back, b_space, b_clear = gr.Button('Backspace'), gr.Button('Add space'), gr.Button('Clear all', variant='stop')
#             cam.stream(on_frame, inputs=[cam, state, record, mode, conf, stable, correct, crop, mirror],
#                        outputs=[out_img, text, status, s1, s2, s3, state], time_limit=180, stream_every=0.2, concurrency_limit=4)
#             for i, b in enumerate([s1, s2, s3]):
#                 b.click(lambda s, i=i: (s.replace_last(i), s.snapshot()['text'], s)[1:], [state], [text, state])
#             b_back.click(lambda s: (s.manual('backspace'), s.snapshot()['text'], s)[1:], [state], [text, state])
#             b_space.click(lambda s: (s.manual('space'), s.snapshot()['text'], s)[1:], [state], [text, state])
#             b_clear.click(lambda s: (s.manual('clear'), s.snapshot()['text'], s)[1:], [state], [text, state])
#         with gr.Tab('Letter from a photo'):
#             gr.Markdown('Upload a photo of one **static** letter sign (works without a webcam).')
#             with gr.Row():
#                 photo = gr.Image(type='numpy', sources=['upload'], label='Photo')
#                 with gr.Column():
#                     pmode = gr.Radio(MODES, value=DEFAULT_MODE, label='Recognition mode')
#                     pcrop = gr.Checkbox(value=False, label='Crop to the hand first')
#                     plabel = gr.Label(num_top_classes=5, label='Most likely letters')
#             photo.change(predict_photo, [photo, pmode, pcrop], plabel)
#         with gr.Tab('Dynamic letter from a video'):
#             gr.Markdown('Upload a short video of one **dynamic** letter (آ ڑ ج ہ).')
#             with gr.Row():
#                 vid = gr.Video(label='Video')
#                 with gr.Column():
#                     vlabel = gr.Label(num_top_classes=4, label='Most likely letters'); vinfo = gr.Markdown()
#             vid.change(predict_video, [vid], [vlabel, vinfo])
#         with gr.Tab('How to use'):
#             gr.Markdown(f"""
# **Control gestures** (hold for about a second): **space** = open flat palm (ends the word and applies dictionary correction), **delete** = thumbs down (removes the last character).

# **Static letters (36):** hold the sign steady in front of the camera until the hold counter reaches its target. To type the same letter twice, drop your hand out of view for a moment in between.

# **Dynamic letters (آ ڑ ج ہ):** tick the *Record* box, perform the motion once, untick the box.

# **Tips:** good light, plain background, hand fully visible. If recognition is poor, try the *Mirror* setting and the *Crop to the hand* setting.

# **Known limits:** about 87-90% of single letters are recognized correctly for new people, so names are often not typed perfectly; use the suggestion buttons to fix a letter. The dictionary only knows about {len(LEX.words)} demo words.
# """)
#             gp = os.path.join(BASE, 'assets', 'control_gestures.jpg')
#             if os.path.exists(gp):
#                 gr.Image(gp, label='Control gestures: open palm = space, thumbs down = delete', interactive=False)

# if __name__ == '__main__':
#     demo.queue().launch()
