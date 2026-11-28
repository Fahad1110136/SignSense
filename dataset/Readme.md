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
| `control_gestures.csv` | `scripts/collect_gestures.py` | 1,634 recorded hand-landmark samples of the **space** (open palm, 3 people) and **delete** (thumbs down, 3 people) gestures. Only landmarks are stored, no images or faces |


 
The processed 224x224 images (about 400 MB) are **regenerated** by `preprocessing.ipynb` and are not committed.
 
## 🔤 Letter names and Urdu characters
 
The class-to-letter table below follows the class numbers in the dataset file names (01 to 40), which follow the Urdu alphabet order. It is used by the typing engine (`scripts/typing_engine.py`).
 
| Class | Urdu | Class | Urdu | Class | Urdu | Class | Urdu |
|---|---|---|---|---|---|---|---|
| Alifmad* | آ | Alif | ا | Bay | ب | Pay | پ |
| Tay | ت | Taay | ٹ | Say | ث | Jeem* | ج |
| Chay | چ | 1-Hay | ح | Khay | خ | Dal | د |
| Daal | ڈ | Zaal | ذ | Ray | ر | Aray* | ڑ |
| Zay | ز | Zaey | ژ | Seen | س | Sheen | ش |
| Suad | ص | Zuad | ض | Tuey | ط | Zuey | ظ |
| Ain | ع | Ghain | غ | Fay | ف | Kaf | ق |
| Kiaf | ک | Gaaf | گ | Lam | ل | Meem | م |
| Nuun | ن | Nuungh | ں | Wao | و | 2-Hay* | ہ |
| Dochahay | ھ | Hamza | ء | Cyeh | ی | Byeh | ے |
 
\*dynamic letters are only recognised from video
 
## 🧪 Splits
 
| Split | Images | Videos | Subjects (images) |
|---|---|---|---|
| Train | 15,062 | 275 | 59 |
| Validation | 3,060 | 59 | 13 |
| Test | 3,238 | 59 | 12 |
 
 