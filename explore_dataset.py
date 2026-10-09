# %% 1. Import libraries and load the data
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    baseDir = Path(__file__).resolve().parent
except NameError:
    baseDir = Path.cwd()

outputDir = baseDir / "outputs"
outputDir.mkdir(exist_ok=True)
df = pd.read_csv(baseDir / "data" / "mnist_train.csv")
testDf = pd.read_csv(baseDir / "data" / "mnist_test.csv")

# %% 2. Basic queries, like the class practice
print("First five rows:")
print(df.head())
print("Training shape:", df.shape)
print("Test shape:", testDf.shape)
print("Column information:")
df.info()
print("Statistics for the label and first five pixels:")
print(df.iloc[:, :6].describe())

# %% 3. Check missing values, duplicates and pixel values
print("Missing values in training data:", df.isnull().sum().sum())
print("Missing values in test data:", testDf.isnull().sum().sum())
print("Duplicate training rows:", df.duplicated().sum())
print("Duplicate test rows:", testDf.duplicated().sum())
pixelData = df.drop(columns="label")
print("Smallest pixel value:", pixelData.min().min())
print("Largest pixel value:", pixelData.max().max())
print("Labels:", sorted(df["label"].unique()))

# %% 4. Ask questions using filters and grouping
print("How many training images belong to each digit?")
labelCounts = df["label"].value_counts().sort_index()
print(labelCounts)
print("How many test images belong to each digit?")
print(testDf["label"].value_counts().sort_index())

digitSeven = df[df["label"] == 7]
print("Number of digit 7 images:", len(digitSeven))
print("First three digit 7 rows:")
print(digitSeven.head(3))
print("Number of rows with labels 0 or 1:", len(df[df["label"].isin([0, 1])]))
print("Most common training digit:", labelCounts.idxmax())
print("Least common training digit:", labelCounts.idxmin())

averageBrightness = pixelData.mean(axis=1)
print("Average pixel intensity by digit (0 is black and 255 is white):")
print(averageBrightness.groupby(df["label"]).mean().round(2))
labelCounts.rename("count").to_csv(outputDir / "label_counts.csv")

# %% 5. Draw a simple chart
plt.figure(figsize=(8, 4))
labelCounts.plot(kind="bar", color="steelblue", rot=0)
plt.title("Number of training images for each digit")
plt.xlabel("Digit")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig(outputDir / "label_counts.png", dpi=150)
plt.show()
plt.close()

# %% 6. Turn pixel rows back into images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in enumerate(axes.flat):
    row = df[df["label"] == digit].iloc[0]
    image = row.drop("label").to_numpy().reshape(28, 28)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Digit {digit}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(outputDir / "sample_digits.png", dpi=150)
plt.show()
plt.close()

# %% 7. Identify the problem and suitable models
print("Target: label, the digit from 0 to 9.")
print("Features: the 784 pixel columns, making a 28 x 28 grayscale image.")
print("Learning type: supervised learning because each image has a label.")
print("Task: multiclass classification because there are ten categories.")
print("A simple first model: K-Nearest Neighbors (KNN).")
print("KNN compares an image with similar labeled images.")
print("A later option: a Convolutional Neural Network (CNN) for image patterns.")
print("Unsupervised K-Means could group images if labels were ignored.")
print("Its groups would not automatically be the correct digit labels.")
print("No model is trained in this exploration.")
