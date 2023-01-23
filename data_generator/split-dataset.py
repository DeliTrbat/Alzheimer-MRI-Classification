
# 60% of files in the dataset will be used for training, 20% for validation, and 20% for testing

import os
import random
import shutil

ORIGINAL_DATASET_DIR = 'OriginalDataset'
SPLIT_DATASET_DIR = 'SplitDataset'


def add_images(to_path: str, from_path: str, image_name_list: list):
    for image in image_name_list:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        shutil.copy(os.path.join(from_path, image), to_path)


def split_data(subfolder: str):
    all_images = os.listdir(os.path.join(ORIGINAL_DATASET_DIR, subfolder))
    random.shuffle(all_images)
    train_end_index = int(0.6 * len(all_images))
    validation_end_index = int(0.8 * len(all_images))
    train_images = all_images[:train_end_index]
    validation_images = all_images[train_end_index:validation_end_index]
    test_images = all_images[validation_end_index:]

    add_images(os.path.join(SPLIT_DATASET_DIR, 'Train', subfolder),
               os.path.join(ORIGINAL_DATASET_DIR, subfolder), train_images)

    add_images(os.path.join(SPLIT_DATASET_DIR, 'Validation', subfolder),
               os.path.join(ORIGINAL_DATASET_DIR, subfolder), validation_images)

    add_images(os.path.join(SPLIT_DATASET_DIR, 'Test', subfolder),
               os.path.join(ORIGINAL_DATASET_DIR, subfolder), test_images)


def main():
    if os.path.exists(SPLIT_DATASET_DIR):
        shutil.rmtree(SPLIT_DATASET_DIR)
    os.mkdir(SPLIT_DATASET_DIR)

    for subfolder in os.listdir(ORIGINAL_DATASET_DIR):
        split_data(subfolder)


if __name__ == '__main__':
    main()
