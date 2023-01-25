import shutil
import random
import numpy as np
import os
import albumentations as A
import cv2
import tqdm

IMAGE_SIZE = 210

iaa_affine_transform = A.Affine(
    shear=(-5, 5),
    scale=(0.75, 1), keep_ratio=True,
    rotate=(-30, 30),
    translate_percent=(0.0, 0.1),
    always_apply=True, cval=0)

brightness_transform = A.RandomBrightnessContrast(
    brightness_limit=0.05, contrast_limit=0.05, always_apply=True)

noise_transform = A.GaussNoise(var_limit=10.0, always_apply=True)

random_flip_transform = A.HorizontalFlip(p=0.5)


transforms = A.Compose(
    [
        iaa_affine_transform,
        random_flip_transform,
        brightness_transform,
        noise_transform,
    ])

ORIGINAL_DATASET_PATH = "SplitDataset"
AUGMENTED_DATASET_PATH = "AugmentedDataset"


def augment_image(from_image_path):
    from_dir_path = os.path.dirname(from_image_path)
    to_dir_path = from_dir_path.replace(
        ORIGINAL_DATASET_PATH, AUGMENTED_DATASET_PATH)
    if not os.path.exists(to_dir_path):
        os.makedirs(to_dir_path)

    for _ in range(10):
        new_file_name = str(random.getrandbits(32)) + ".jpg"
        to_image_path = os.path.join(to_dir_path, new_file_name)

        image = cv2.imread(from_image_path)
        data = {"image": image}
        data_aug = transforms(**data)
        image_aug = data_aug["image"]

        cv2.imwrite(to_image_path, image_aug)


BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} files'


def augment_folder_recursive(folder_path):
    directories, files = [], []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            directories.append(entry.path)
        else:
            files.append(entry.path)

    for directories in directories:
        augment_folder_recursive(directories)

    if len(files) == 0:
        return

    pbar = tqdm.tqdm(
        total=len(files), desc="Augmenting... " +
        folder_path, bar_format=BAR_FORMAT
    )
    for file in files:
        augment_image(file)
        pbar.update(1)

    pbar.close()


def main():
    if os.path.exists(AUGMENTED_DATASET_PATH):
        shutil.rmtree(AUGMENTED_DATASET_PATH)
    augment_folder_recursive(ORIGINAL_DATASET_PATH)


if __name__ == "__main__":
    main()
