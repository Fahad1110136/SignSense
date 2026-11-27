# SignSense web app

Live Urdu fingerspelling to text (Gradio). Webcam typing, photo and video upload tabs.

## Folder contents
```
app/
├── app.py                 Gradio interface
├── session.py             per-user logic (frames -> events -> typing engine)
├── recognizer.py          MediaPipe landmarks + the trained models
├── typing_engine.py       letter buffer, stability check, space/delete, dictionary correction
├── app_config.json        settings written by notebooks/app_prep.ipynb
├── urdu_lexicon.txt       demo dictionary
├── requirements.txt
├── assets/control_gestures.jpg
└── models/                baseline_cnn_fp32.keras, mobilenetv2_fp32.keras, efficientnetb0_fp32.keras,
                           control_gesture_clf.joblib, dynamic_rf.joblib
```

## Run locally
```
cd app
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860 and allow the camera. The first start downloads the 7 MB MediaPipe hand model.
The tabs *Letter from a photo* and *Dynamic letter from a video* work without a webcam.

## How the live tab works
1. MediaPipe finds the hand and gives 21 landmarks.
2. A small classifier on the landmarks recognizes the control gestures (open palm = space, thumbs down = delete).
3. Otherwise the hand image is classified by the static letter models (3-model ensemble, or MobileNetV2 alone in *Fast* mode).
4. Dynamic letters are recorded with the *Record* checkbox and classified by the Random Forest on the landmark sequence.
5. The typing engine commits a letter only after it was held steadily, then corrects finished words with the dictionary.