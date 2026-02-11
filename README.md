# Face Mask Detection Project

A Deep Learning-based system to detect face masks in real-time or from images, built using TensorFlow, Keras, and OpenCV.

## 🚀 Features
- **Image Processing**: Detect masks on multiple images in a folder.
- **Real-time Detection**: Live video stream mask detection via webcam.
- **High Accuracy**: Uses a MobileNetV2 architecture for mask classification.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd facerecognition
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment or Python 3.7 - 3.11.
   ```bash
   pip install -r requirements.txt
   ```

---

## 📸 Running on Images (Recommended for WSL/Remote)

If you don't have a webcam or are using WSL, use the image detection script. It processes images from the `dataset` folder and saves results to `output_results/`.

```bash
python detect_mask_image.py -n 5
```

- `-n`: Number of images to process (default is 10).
- `-i`: Input directory (default is `dataset`).
- `-o`: Output directory (default is `output_results`).

---

## 📹 Running on Live Video (Webcam)

To start the real-time detector using your computer's webcam:

```bash
python detect_mask_video.py
```

- Clear frames will appear with labels: **Mask** (Green) or **No Mask** (Red).
- Press **'q'** to exit the video stream.

---

## 🧠 Training the Model (Optional)

If you want to re-train the mask detector on your own dataset:

```bash
python train_mask_detector.py --dataset dataset
```

---

## 📂 Project Structure
- `dataset/`: Contains images for training/testing.
- `face_detector/`: Pre-trained Caffe model for face localization.
- `mask_detector.h5`: The trained face mask classification model.
- `detect_mask_image.py`: Script for processing static images.
- `detect_mask_video.py`: Script for real-time video stream detection.
