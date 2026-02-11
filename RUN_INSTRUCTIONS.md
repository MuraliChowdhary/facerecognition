# How to Run the Face Mask Detection Project

Follow these steps to set up and run the Face Mask Detection project on your local machine.

## Prerequisites

- **Python**: Ensure you have Python installed (version 3.7 to 3.10 recommended).
- **Webcam**: A functional webcam is required for real-time detection.

## Step-by-Step Instructions

### 1. Open the Project in Terminal
Navigate to the project directory where the files are located (e.g., `c:\Users\Ramakrishna\rkface\Face-Mask-Detection-master\Face-Mask-Detection-master`).

### 2. Install Dependencies
Install the required Python libraries using the `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the Mask Detector or Skip to step 4
If you want to re-train the model using the dataset provided in the `dataset` folder:

```bash
python train_mask_detector.py
```
*Note: This process may take some time depending on your computer's performance. A `mask_detector.model` file will be created/updated upon completion.*

### 4. Run the Face Mask Detector
To start the real-time video stream and detect face masks:

```bash
python detect_mask_video.py
```

- A window named "Frame" will appear showing the video feed.
- The system will detect faces and label them as **Mask** (Green) or **No Mask** (Red) along with the confidence score.

### 5. Stop the Application
To exit the video stream, press the `q` key on your keyboard while the video window is active.
