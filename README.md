# Driver Drowsiness Detection (CNN + MediaPipe)

Real-time driver drowsiness detection. A CNN classifies each eye as **open** or **closed**, and MediaPipe Face Mesh locates the eyes in the webcam image. If both eyes stay closed longer than a set time (default 1.5 s), an alarm is raised.

## How it works

1. **Training** (`train_model.py`): trains a small CNN on grayscale eye images (open vs. closed) and evaluates it on a held-out test set.
2. **Detection** (`realtime_detection_cnn.py`): for each webcam frame, MediaPipe Face Mesh finds both eyes, each eye is cropped, resized and classified by the CNN, and the average P(closed) is compared with a threshold. A timer triggers the alert and alarm sound.

The positive class is **closed** (label 1), because missing a closed eye is the costly mistake. Recall therefore means: of all truly closed eyes, how many were caught.

## Project structure

```
.
├── train_model.py                 # train + evaluate the CNN
├── realtime_detection_cnn.py      # local real-time webcam detection
├── requirements.txt
└── README.md
```

Generated when you train (not committed to git):

```
data/                  # your dataset
outputs/
├── eye_state_cnn.keras        # trained model
├── metrics.json               # accuracy, precision, recall, F1 (test set)
├── classification_report.txt
├── confusion_matrix.png
└── training_curves.png
```

## Setup

Python 3.9–3.11 is recommended (MediaPipe 0.10.14 has no wheels for newer versions).

```bash
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download an eye-state dataset such as the [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) or a Kaggle drowsiness eye dataset, and arrange it like this:

```
data/
├── closed/   img1.png, img2.png, ...
└── open/     img1.png, img2.png, ...
```

## 1. Train the model

```bash
python train_model.py --data-dir data --epochs 15
```

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Folder containing `open/` and `closed/` |
| `--out-dir` | `outputs` | Where the model and results are saved |
| `--img-size` | `64` | Eye image size (pixels) |
| `--batch-size` | `64` | Batch size |
| `--epochs` | `15` | Maximum epochs (early stopping is on) |
| `--max-per-class` | `0` | Limit images per class (0 = all), useful for quick tests |

The data is split 70% train / 15% validation / 15% test (stratified). The training script reports test-set accuracy, precision, recall and F1, and saves a confusion matrix and training curves.

## 2. Run real-time detection (local webcam)

```bash
python realtime_detection_cnn.py
python realtime_detection_cnn.py --model outputs/eye_state_cnn.keras --seconds 1.5
```

| Option | Default | Description |
|---|---|---|
| `--model` | `outputs/eye_state_cnn.keras` | Path to the trained model |
| `--camera` | `0` | Camera index |
| `--seconds` | `1.5` | Eyes closed this long triggers the alert |
| `--threshold` | `0.5` | P(closed) above this counts as a closed eye |
| `--no-sound` | off | Disable the alarm sound |

Keys: `q` or `Esc` to quit, `r` to reset the alert counter.

## Google Colab

Colab cannot access your local webcam through OpenCV, so `colab_webcam_after_training.py` streams frames from the browser camera instead.

1. Run the training cells in the notebook so `outputs/eye_state_cnn.keras` exists.
2. Paste the cells W1–W4 from `colab_webcam_after_training.py` into new cells and run them in order.
3. Click **Start camera**, allow camera access, and click **Stop** to finish.

Expect about 3–10 fps and some lag, because each frame is sent to the Colab server and back. For smooth detection, use the local script. Use Chrome or Edge.

## Troubleshooting

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: newer MediaPipe removed the old API. Run `pip install mediapipe==0.10.14`.
- **Could not open camera**: try `--camera 1`, and close other apps that use the webcam.
- **Open eyes detected as closed (or the reverse)**: watch the P(closed) value on screen and adjust `--threshold`. Training images (often close-up or infrared) can differ from webcam crops.
- **Wrong labels**: the script assumes closed = 1 and open = 0, as in `train_model.py`.

## Results

Evaluated on a held-out test set of 26,214 images (15% of the data, stratified). Positive class = closed eyes.

Dataset: 133,103 open and 41,653 closed eye images. Split: 122,329 train / 26,213 validation / 26,214 test. Model: 101,889 parameters.

| Metric | Value |
|---|---|
| Accuracy | 99.05% |
| Precision | 97.77% |
| Recall | 98.26% |
| F1-score | 98.01% |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted open | Predicted closed |
|---|---|---|
| **Actual open** | 19,826 | 140 |
| **Actual closed** | 109 | 6,139 |

The classes are imbalanced (about 76% open), so precision, recall and F1 for the closed class are more informative than accuracy alone. These numbers come from a random image-level split; real webcam performance can be lower (see Limitations).

## Limitations

- The model only looks at eye state, not yawning or head pose.
- Accuracy depends on lighting, camera angle, glasses and the training data.
- The train/validation/test split is random per image, so images of the same person may appear in more than one split. This can make the test scores optimistic compared with unseen people and real webcams.
- This is a learning and research project and **must not be relied on for real driver safety**.

