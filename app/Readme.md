# SignSense App

The SignSense web app spell a word in Urdu Sign Language in front of a webcam and watch it appear as Urdu text. It is built with **Gradio** and runs on a normal laptop CPU, with no GPU and no internet connection needed after the first start.

## Run

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860 in a browser and allow camera access. The first start takes up to a minute while TensorFlow loads and the three letter models warm up.

Notes:

- Use a virtual environment (`python -m venv venv`) to keep the large TensorFlow install separate.
- The camera only works on `localhost` or over HTTPS, which is the case for the address above.
- `hand_landmarker.task` (7.8 MB) is included. If it is missing, the app downloads it once from Google, which needs an internet connection.
- The Urdu font (Noto Nastaliq Urdu) is loaded from Google Fonts. Without internet the text is shown in a fallback font and is still correct.

## Tabs

| Tab | What it does |
|---|---|
| Live typing (webcam) | Types Urdu text from the webcam: static letters, control gestures, dynamic letters via a record box, top-3 suggestions for the last letter |
| Letter from a photo | Upload a photo of one static letter and see the five most likely letters (works without a webcam) |
| Dynamic letter from a video | Upload a short video of one dynamic letter and see the four class probabilities |
| How to use | Short instructions and a picture of the two control gestures |

## Using the live tab

- **Static letters (36):** hold the sign steady until the hold counter in the status box reaches its target. To type the same letter twice, drop your hand out of view for a moment in between.
- **Space:** hold an open flat palm. This ends the word and applies dictionary correction.
- **Delete:** hold a thumbs down. This removes the last character.
- **Dynamic letters (آ ڑ ج ہ):** tick the record box, perform the motion once, then untick it.
- **Wrong letter:** click one of the three suggestion buttons to replace the last letter with the next best guess.
- **Buttons:** Backspace, Add space and Clear all do the same jobs as the gestures, so the app can be used without them.

### Settings

| Setting | Default | Effect |
|---|---|---|
| Recognition mode | Accuracy | Accuracy averages the three static models (222 ms per letter prediction on a CPU, test accuracy 89.8%). Fast uses MobileNetV2 only (55 ms, test accuracy 87.4%) |
| Minimum confidence of a letter | 0.5 | A frame below this confidence never counts towards a letter |
| Frames a letter must be held | 3 (accuracy) or 5 (fast) | More frames means fewer accidental letters but slower typing |
| Dictionary correction | on | Replaces a finished word by the single closest dictionary word if it is at most one edit away |
| Crop to the hand | on | Cuts the image around the detected hand before the letter models see it |
| Mirror the camera image | off | Try this if letters are not recognized |

## Files in this folder

| File | Description |
|---|---|
| `app.py` | The Gradio interface and the event handlers of the four tabs |
| `session.py` | One typing session per browser tab: receives a frame, asks the recognizer what it sees, feeds the result to the typing engine, and builds the text and status shown in the interface. No TensorFlow or Gradio imports |
| `recognizer.py` | Loads and runs the models: MediaPipe hand landmarks, control-gesture classifier, the three static letter models and the dynamic Random Forest |
| `typing_engine.py` | Turns per-frame events into text: stability check, neutral gap before a repeated letter, space and delete, dictionary correction (a copy of `scripts/typing_engine.py`) |
| `light_models.py` | Retrains the two small scikit-learn models from `data/` if the saved ones cannot be loaded |
| `app_config.json` | Settings read at start-up: model files, padding colour, control threshold, typing-engine defaults, lexicon file (see [`models/Readme.md`](../models/Readme.md)) |
| `urdu_lexicon.txt` | Demo dictionary of 170 Urdu words used for correction |
| `hand_landmarker.task` | MediaPipe hand-landmark model |
| `requirements.txt` | Python packages |
| `models/` | The five models the app loads: `baseline_cnn_fp32.keras`, `mobilenetv2_fp32.keras`, `efficientnetb0_fp32.keras`, `dynamic_rf.joblib`, `control_gesture_clf.joblib` |
| `data/` | Landmark data used only for retraining the two small models: `control_gestures.csv`, `control_negatives.npz`, `dynamic_landmarks.npz` |
| `assets/` | `control_gestures.jpg`, the picture shown in the How to use tab |

## How one frame is processed

1. The browser sends a webcam frame about every 0.2 seconds. If the previous frame of the same user is still being processed, the new one is skipped.
2. MediaPipe finds the hand and returns 21 landmarks. With no hand, the frame counts as the neutral position.
3. The control-gesture classifier checks the landmarks for space and delete (threshold 0.83). Control gestures take priority over letters.
4. Otherwise the hand is cropped, padded to a square, resized to 224x224 and passed to the static letter models.
5. The typing engine commits a letter only after it was the top prediction above the confidence setting for the required number of frames in a row. Space and delete need 8 frames.
6. For dynamic letters, the landmarks are collected while the record box is ticked, resampled to 88 frames and classified by the Random Forest when the box is unticked. At least 12 frames with a visible hand are needed.

Each browser tab has its own session, so several people can type at the same time (up to four frames are processed in parallel).

## Troubleshooting

| Problem | What to do |
|---|---|
| `TypeError: Image.__init__() got an unexpected keyword argument 'mirror_webcam'` | Caused by Gradio 6. `app.py` detects the installed Gradio version and uses the right option, so update to the current `app.py` |
| `InconsistentVersionWarning` about `DecisionTreeClassifier` or `RandomForestClassifier` | The saved `.joblib` files were made with scikit-learn 1.6.1. Run `pip install scikit-learn==1.6.1`. If loading fails anyway, the app retrains the two models from `data/` automatically |
| Letters are not recognized or are always wrong | Tick the mirror setting, improve the lighting, use a plain background and keep the whole hand in the picture |
| Typing feels slow | Switch to Fast mode, or lower the number of frames a letter must be held |
| No camera picture | Allow camera access in the browser and close other programs that use the camera |
| TensorFlow messages about oneDNN, XNNPACK or the feedback manager at start-up | Harmless. Set `TF_ENABLE_ONEDNN_OPTS=0` to hide the oneDNN one |

## Limitations

- About 87 to 90% of single static letters are recognized correctly for new people, so names that are not in the dictionary are often not typed perfectly. The suggestion buttons help.
- Dynamic letters need the record box. Automatic switching between still and moving letters was tested and did not work reliably.
- The dictionary is a small demo list.
- Accuracy depends on lighting, background and camera, because the models were trained on dataset-style photos.

More detail on the models and their evaluation is in [`models/Readme.md`](../models/Readme.md) and the main [`Readme.md`](../Readme.md).