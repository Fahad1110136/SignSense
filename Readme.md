# SignSense: Urdu Sign Language Recognition and Text Generation

This repository holds **Phase 1** (data cleaning, exploratory analysis, preprocessing) of our project. No models are trained yet.

## Project description

SignSense aims to recognise Urdu Sign Language (USL) letters, both static hand shapes and short motion-based signs, and turn a sequence of signed letters into written Urdu words in real time, like a keyboard turning keystrokes into text.

## Problem statement

Sign language is the main way the deaf and hard-of-hearing community in Pakistan communicates, yet USL is understudied compared to languages like ASL, and most existing work stops at recognising static alphabet images without producing readable text. Names and words without a dedicated sign are communicated by fingerspelling, letter by letter, so letter-level recognition is a necessary building block.

The full project will:
1. classify the 36 static USL letters from images (CNN with transfer learning),
2. classify the 4 dynamic letters from video (MediaPipe hand landmarks + LSTM),
3. combine both in a real-time pipeline that builds Urdu words, with a space gesture to end a word and a delete gesture to correct mistakes.

## Dataset

| | |
|---|---|
| **Name** | Pakistan Sign Language Urdu Alphabets (UAlpha40) |
| **Source** | Bahria University, Karachi Campus, on Mendeley Data |
| **Link** | https://data.mendeley.com/datasets/3pvnnckxyb/1 (DOI 10.17632/3pvnnckxyb.1) |
| **License** | CC BY 4.0 |
| **Size** | about 5 GB; 22,865 JPG images (36 static classes) and 393 MP4 videos (4 dynamic classes) |
| **Features** | the image pixels / video frames, plus metadata we derive: subject id, original or generated copy, width, height, file size, brightness, contrast, blur (images); frame count, fps, duration (videos) |
| **Target variable** | the letter class: 40 classes, given by the folder name |

The dataset is too large for GitHub, so it is **not** included here. The notebooks download it automatically into Google Colab. (Our proposal quoted about 22,280 images; the actual download contains 22,865.)

Static classes: 1-Hay, Ain, Alif, Bay, Byeh, Chay, Cyeh, Daal, Dal, Dochahay, Fay, Gaaf, Ghain, Hamza, Kaf, Khay, Kiaf, Lam, Meem, Nuun, Nuungh, Pay, Ray, Say, Seen, Sheen, Suad, Taay, Tay, Tuey, Wao, Zaal, Zaey, Zay, Zuad, Zuey.  
Dynamic classes: 2-Hay, Alifmad, Aray, Jeem.

## Summary of cleaning steps

Cleaning is non-destructive: raw files are never changed, and `dataset/clean_metadata.csv` lists exactly which files are kept. Details are in `notebooks/data_cleaning.ipynb`.

| Issue | Files affected | Action |
|---|---|---|
| Zero-byte or unreadable (corrupt) images / videos | 0 | none needed |
| Mixed-case extensions (`.JPG` vs `.jpg`) | 1,678 | normalised in metadata, files untouched |
| Exact duplicate images (identical bytes) | 1,485 | removed, keeping the properly named copy |
| Mislabeled copies (20 `Gaaf` images stored in the `Ghain` folder) | 20 | removed from `Ghain` |
| `Gaaf` folder had 1,152 files, about double the other classes, because a set was duplicated | 584 duplicates | reduced to 568 after removing duplicates |
| Near-duplicates (same perceptual hash within a class) | 6,674 | flagged, kept (expected: augmented copies) |
| Size, brightness or blur outliers (IQR rule per class) | 5,586 | flagged, kept (unusual images are still valid signs) |
| Near-blank images | 0 | none needed |
| Video frame-count outliers | 2 | flagged, kept |

**Result:** 22,865 images became **21,360**, and all **393 videos** were kept (none corrupt, none duplicated).

## Preprocessing summary

Details are in `notebooks/preprocessing.ipynb`.

- **Resize:** pad to a 224x224 square (keeps the hand undistorted, border filled with the dataset mean colour).
- **Split:** train / validation / test = 70 / 15 / 15, **by subject** so augmented copies of one person's photos never appear in two splits (59 / 13 / 12 subjects; 15,062 / 3,060 / 3,238 images). Videos have no subject id in their filenames, so they use a stratified random split (about 61-74 train, 13-16 validation, 13-16 test videos per class).
- **Normalization:** per-channel mean (0.689, 0.625, 0.601) and std (0.211, 0.230, 0.243), computed on the training split only; after normalization the mean is about 0 and the std about 1.
- **Class weights:** balanced weights from the training split (static 0.87 to 1.12, dynamic 0.93 to 1.13).
- **Dynamic signs:** each video is resampled to 88 frames, MediaPipe extracts 21 hand landmarks per frame, and landmarks are normalised for hand position and size. The result is a (393, 88, 63) array.

## Key findings

- **Static classes are balanced:** 533 to 679 images per class (imbalance ratio 1.27), so accuracy plus F1 is a fair metric for the static model.
- **Dynamic classes are small:** 87 to 105 videos each, about 6 times fewer samples than a static class, so the sequence model will need augmentation and a small architecture.
- **Most images are augmented copies:** about 86% are generated; only 2,897 originals exist across 84 subjects. A random image-level split would leak, which is why we split by subject. We can also report test results on original images only.
- **Images vary widely:** widths of 204 to 2,106 px, heights of 226 to 2,414 px, and mixed portrait / landscape orientation, so a fixed-size padded input is needed.
- **The data is brighter than ImageNet** (mean pixel about 0.62 to 0.69 versus 0.41 to 0.49), so we use dataset-specific normalization rather than ImageNet's.
- **Videos are short and uneven:** all at 30 fps, 13 to 118 frames (0.4 to 3.9 s), so a fixed sequence length of 88 frames is used.
- **Hand landmarks work well on the videos:** MediaPipe found a hand in 97.1% of frames, and no video has under half of its frames detected.
- More detail (letter similarity, whether image properties depend on the class, hand detection on static images) is in `notebooks/eda.ipynb`, and every figure is explained in `images/Readme.md`.

## Repository structure

```
SignSense/
├── Readme.md
├── requirements.txt
├── dataset/       small data files and metadata (see dataset/Readme.md)
├── notebooks/     setup_and_data_inventory, data_cleaning, eda, preprocessing
└── images/        all saved figures with a short note on each (see images/Readme.md)
```

## How to run the notebooks

The data is large, so run everything in **Google Colab** (a GPU is not needed in Phase 1).

1. Open Colab and upload a notebook from `notebooks/` (File, Upload notebook).
2. Run the notebooks **in this order**, using Runtime, Run all:

| # | Notebook | What it does | Needs uploaded | Approx. time |
|---|---|---|---|---|
| 1 | `setup_and_data_inventory.ipynb` | downloads the dataset from Mendeley, inventories files | nothing | 10 min |
| 2 | `data_cleaning.ipynb` | cleaning; produces `clean_metadata.csv` | nothing | 10-20 min |
| 3 | `eda.ipynb` | exploratory analysis and figures | `dataset/clean_metadata.csv` | 10-15 min |
| 4 | `preprocessing.ipynb` | split, resize, normalization, landmark sequences | `dataset/clean_metadata.csv` and `dataset/channel_stats.json` | 25-40 min |

3. Each notebook downloads the dataset again if the Colab session was reset, so run them within one session where possible.
4. Outputs appear in the Colab folder `outputs/`; download them and place them in `dataset/` and `images/`.

To install the libraries locally instead: `pip install -r requirements.txt`.

## Next phase

Train the static classifier (MobileNetV2 or EfficientNet-B0) on `processed_metadata.csv`, train the LSTM on `dynamic_landmarks.npz`, and build the real-time letter-to-word pipeline.