import gdown
import zipfile

url = "https://drive.google.com/u/0/uc?id=1x6v02fTqE0SSumZgX65DSi9ijSkRZX6d&export=download"
output = "OriginalDataset.zip"
gdown.download(url, output)

with zipfile.ZipFile("OriginalDataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./")

exec(open("split-dataset.py").read())

exec(open("albumentation-dataset.py").read())
