import gdown
import zipfile
import shutil

from split_dataset import main as split_dataset
from albumentation_dataset import main as albumentation_dataset

url = "https://drive.google.com/u/0/uc?id=1x6v02fTqE0SSumZgX65DSi9ijSkRZX6d&export=download"
output = "OriginalDataset.zip"
gdown.download(url, output)
print("\n")

print("Extracting dataset...")
with zipfile.ZipFile("OriginalDataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./")
print("\n")

print("Splitting dataset...")
split_dataset()
print("\n")

print("Deleting original dataset...")
shutil.rmtree("OriginalDataset")

print("Creating albumentation dataset...")
albumentation_dataset()
print("\n")

print("Deleting split dataset...")
shutil.rmtree("SplitDataset")
