# Dataset

**Pakistan Sign Language Urdu Alphabets** (UAlpha40), Bahria University, Karachi Campus  
Authors: Shahood Sajid, Sameena Javaid, Yusra Baloch | License: CC BY 4.0  
Download: https://data.mendeley.com/datasets/3pvnnckxyb/1 (DOI 10.17632/3pvnnckxyb.1)

The raw data is about 5 GB and is **not stored in this repository**. The notebooks download it directly into Google Colab.

## Files in this folder

| File | Produced by | Description |
|---|---|---|
| `file_inventory.csv` | `setup_and_data_inventory.ipynb` | Every raw file with folder, extension and size |
| `clean_metadata.csv` | `data_cleaning.ipynb` | One row per kept file (21,360 images + 393 videos) with label, subject, original/generated flag, size, brightness, contrast, blur and video statistics |
| `cleaning_summary.csv` | `data_cleaning.ipynb` | Each cleaning issue, how many files it affected, and the action taken |
| `channel_stats.json` | `eda.ipynb` | RGB mean/std from a sample of all cleaned images (used for the padding colour) |
| `processed_metadata.csv` | `preprocessing.ipynb` | Final table for modeling: processed path, class id, subject and train/val/test split |
| `split_summary.csv` | `preprocessing.ipynb` | Images or videos per class in each split |
| `label_map.json` | `preprocessing.ipynb` | Class name to id (static ids 0-35, dynamic ids 36-39) |
| `class_weights.json` | `preprocessing.ipynb` | Balanced class weights from the training split |
| `channel_stats_train.json` | `preprocessing.ipynb` | RGB mean/std of the training split (used for normalization) |
| `dynamic_landmarks.npz` | `preprocessing.ipynb` | MediaPipe hand-landmark sequences for the 393 videos: array `X` (393, 88, 63), detection `mask`, labels `y`, `split`, `files` |
