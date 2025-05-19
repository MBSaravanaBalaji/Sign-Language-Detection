<p align="center">
  <img src="working detection.png" alt=" Test Image" width="400px"/>
</p>

# Sign Language Recognition

An end-to-end system for recognizing American Sign Language (ASL) gestures in real-time using Mediapipe Holistic for landmark detection and an LSTM neural network for temporal sequence classification.

---

## Overview

This project demonstrates a complete pipeline for ASL gesture recognition:

1. **Data Collection**: Capture keypoints for predefined gestures via your webcam.
2. **Preprocessing**: Organize and vectorize landmark sequences.
3. **Model Training**: Train a stacked LSTM network to classify sequences of landmarks.
4. **Real-Time Inference**: Run live webcam detection with dynamic sentence overlay.

By leveraging Mediapipe for robust pose and hand tracking, combined with an LSTM that models temporal dynamics, this system achieves smooth, accurate recognition in real time.

---

## Features

* 🎥 **Webcam Integration**: Stream live video and extract pose & hand landmarks.
* 🔑 **Keypoint Extraction**: Save landmark data (`.npy`) for each gesture sample.
* 📊 **LSTM Classifier**: Two-layer LSTM network with dropout and dense layers.
* ✅ **Real-Time Inference**: Display recognized gestures as a running sentence overlay.
* 🔒 **Threshold Filtering**: Confidence-based filtering to reduce misclassifications.

---


## Installation



2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```

3. **Install dependencies**


   ```bash
   pip install opencv-python mediapipe tensorflow numpy
   ```

---

## Project Structure

```plaintext
├── Sign-lang.ipynb       # Jupyter notebook: data collection ➔ preprocessing ➔ training
├── actions.h5            # Saved LSTM model weights
├── keypoints/            # Folders with .npy keypoint files per action
├── Sign-lang.py # 
└── requirements.txt      # Python dependencies
```

---

## Data Collection

1. Open `Sign-lang.ipynb`.
2. Define your list of gesture labels (`actions = [ ... ]`).
3. Run the cell that creates directories under `keypoints/<action>/<sample>`.
4. Execute the loop to record 30 frames per sample. Each sample saves a `.npy` of concatenated keypoints

---

## Preprocessing

* Load all `.npy` files into a feature array `X` and label array `y`.
* One-hot encode `y` for Keras compatibility.
* Split into training and testing sets (e.g., 80/20).

---

## Model Training

* **Architecture**:

  * LSTM (64 units) with `return_sequences=True`
  * Dropout (20%)
  * LSTM (128 units)
  * Dense (64) ➔ Dense (`len(actions)`, softmax)

* **Hyperparameters**:

  * Optimizer: `Adam`
  * Loss: `categorical_crossentropy`
  * Epochs: 2000
  * Batch size: 32

* **Usage**:

  1. Run the notebook cell to build and compile the model.
  2. Train on `X_train` / `y_train` and validate on `X_test` / `y_test`.
  3. Save weights to `actions.h5`.

---

**Key parameters**:

* `threshold` (default: 0.4): Confidence threshold to accept a prediction
* `sequence_length`: Number of frames buffered (default: 30)

Your webcam feed will open, display landmarks, and overlay recognized gestures as a sentence at the top.

---
