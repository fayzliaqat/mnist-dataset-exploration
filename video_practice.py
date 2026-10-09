from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd

# With no input, make a short slideshow from ten labeled MNIST images.
# This is only a drawing exercise, not a trained digit detector.
baseDir = Path(__file__).resolve().parent
outputDir = baseDir / "outputs"
outputDir.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description="Draw a box and text on a video.")
parser.add_argument("--input", type=Path, help="Optional path to your own video")
args = parser.parse_args()

inputPath = args.input
if inputPath is None:
    inputPath = outputDir / "input_demo.mp4"
    df = pd.read_csv(baseDir / "data" / "mnist_train.csv")
    writer = cv2.VideoWriter(str(inputPath), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Could not create the input video.")
    try:
        for digit in range(10):
            pixels = df[df["label"] == digit].iloc[0].drop("label").to_numpy(dtype=np.uint8)
            image = cv2.resize(pixels.reshape(28, 28), (280, 280), interpolation=cv2.INTER_NEAREST)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[100:380, 180:460] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for _ in range(10):
                writer.write(frame)
    finally:
        writer.release()

cap = cv2.VideoCapture(str(inputPath))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not read video: {inputPath}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if width <= 0 or height <= 0 or fps <= 0:
    cap.release()
    raise ValueError("The video has invalid dimensions or frame rate.")
outputPath = outputDir / "week1_output.mp4"
if inputPath.resolve() == outputPath.resolve():
    cap.release()
    raise ValueError("Choose an input other than week1_output.mp4.")
writer = cv2.VideoWriter(str(outputPath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

frameCount = 0
try:
    while True:
        success, frame = cap.read()
        if not success:
            break
        # A fixed box marks the center area. It does not detect an object.
        cv2.rectangle(frame, (int(width * 0.25), int(height * 0.18)),
                      (int(width * 0.75), int(height * 0.82)), (0, 255, 0), 2)
        cv2.putText(frame, "Week 1 - OpenCV drawing practice", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frameCount + 1}", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
        frameCount += 1
finally:
    cap.release()
    writer.release()
if frameCount == 0:
    raise ValueError("The input video contains no readable frames.")
print(f"Saved {frameCount} frames to {outputPath}")
