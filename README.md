# GestureOS — Deep Learning Hand Gesture Recognition

GestureOS is a Windows/Python starter project for collecting hand-landmark data, training a TensorFlow/Keras neural network, and recognizing gestures in real time through a webcam.

## Requirements

- Windows 10/11
- Python 3.12 (64-bit)
- VS Code (recommended)
- Webcam
- Internet connection for first-time package installation

## Setup

1. Extract the ZIP file.
2. Open the extracted `GestureOS` folder in VS Code.
3. Double-click `setup.bat`, or open a PowerShell terminal in this folder and run:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   and then activate the environment again.

## Step 1 — Collect data

Run `collect_data.bat`, or:

```powershell
.\.venv\Scripts\Activate.ps1
python collect_data.py
```

- Press `1` for `Open_Palm`
- Press `2` for `Fist`
- Press `3` for `Peace`
- Press `4` for `Thumbs_Up`
- Press `5` for `Stop`
- Press SPACE to save one sample for the selected gesture
- Press Q to quit

Aim for **100–300 or more varied samples per class** to start. Change the angle, distance, and lighting gradually. Keep labels consistent. The Open_Palm and Stop classes may look similar; if you cannot define a clear visual difference, replace one class consistently before collecting data.

Samples are stored in `dataset/landmarks.csv`. The collector appends to the existing file, so avoid mixing incompatible label definitions in the same dataset.

## Step 2 — Train the model

After collecting samples for every class, run:

```powershell
python train_model.py
```

The script creates:
- `models/gesture_model.keras`
- `models/labels.json`
- `reports/evaluation.txt`
- `reports/confusion_matrix.png`
- `reports/training_history.png`

The evaluation uses a held-out test split. Because nearby webcam frames can be very similar, this split is an initial evaluation rather than proof of generalization to new people. For a stronger case study, collect separate test sessions or test data from people not represented in training.

## Step 3 — Run live recognition

```powershell
python gesture_recognition.py
```

Press Q to quit.

## How it works

- MediaPipe extracts 21 hand landmarks.
- The 21 x/y/z coordinates are made wrist-relative and scale-normalized.
- A trained fully connected neural network (MLP) predicts the gesture class.
- The application displays the predicted label and softmax score.

MediaPipe is used for landmark extraction; the Keras neural network is the model trained for this project.

## Case study experiments

Compare your measured model results with a baseline, such as a rule-based landmark classifier. Report test accuracy, precision, recall, F1-score, confusion matrix, inference latency, and FPS. Record actual values from your own runs; do not claim results before testing.

## Troubleshooting

- **Webcam not opening:** Close other camera apps and check Windows Settings → Privacy & security → Camera.
- **`py` not recognized:** Reinstall Python 3.12 (64-bit) and enable the Python launcher / PATH option, or use `python` if it points to Python 3.12.
- **Package installation fails:** Check that the terminal is using the `.venv` environment and that the internet connection works. Read the first pip error; don't randomly upgrade individual packages.
- **Model file missing:** Collect data and run training before starting live recognition.
