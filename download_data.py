from pathlib import Path
from urllib.request import urlopen
import zipfile

# Download the public Kaggle dataset. No API key is stored in this project.
baseDir = Path(__file__).resolve().parent
dataDir = baseDir / "data"
dataDir.mkdir(exist_ok=True)
url = "https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv"
zipPath = dataDir / "mnist.zip"

if all((dataDir / name).exists() for name in ["mnist_train.csv", "mnist_test.csv"]):
    print("The dataset is already downloaded.")
else:
    print("Downloading MNIST in CSV from Kaggle...")
    with urlopen(url, timeout=180) as response:
        zipPath.write_bytes(response.read())
    with zipfile.ZipFile(zipPath) as archive:
        # Extract only the two expected files.
        for name in ["mnist_train.csv", "mnist_test.csv"]:
            (dataDir / name).write_bytes(archive.read(name))
    print("Saved mnist_train.csv and mnist_test.csv in data/.")
