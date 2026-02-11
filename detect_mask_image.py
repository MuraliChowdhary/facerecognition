#!/usr/bin/env python3
"""
Face Mask Detection on Images
Processes images from the dataset folder instead of using webcam
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import glob

def detect_and_predict_mask(frame, faceNet, maskNet):
    """Detect faces and predict mask/no-mask"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)

def process_image(image_path, faceNet, maskNet, output_dir):
    """Process a single image and save result"""
    print(f"[INFO] Processing: {os.path.basename(image_path)}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    
    orig = image.copy()
    (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
    
    if len(locs) == 0:
        print(f"  -> No faces detected")
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        print(f"  -> Detected: {label}")
    
    # Save the output image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"  -> Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="dataset",
                        help="path to input directory containing images")
    parser.add_argument("-o", "--output", type=str, default="output_results",
                        help="path to output directory for processed images")
    parser.add_argument("-n", "--num", type=int, default=10,
                        help="number of images to process (default: 10)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("[INFO] Loading face detector model...")
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    print("[INFO] Loading face mask detector model...")
    maskNet = load_model("mask_detector.h5")
    
    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {args.input}")
        return
    
    print(f"[INFO] Found {len(image_paths)} images")
    print(f"[INFO] Processing {min(args.num, len(image_paths))} images...\n")
    
    # Process images
    for i, image_path in enumerate(image_paths[:args.num]):
        process_image(image_path, faceNet, maskNet, args.output)
        print()
    
    print(f"[INFO] Done! Results saved to: {args.output}")
    print(f"[INFO] View the output images to see the detection results")

if __name__ == "__main__":
    main()
