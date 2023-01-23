import albumentations as A
import cv2

IMAGE_SIZE = 210

# Open image
image = cv2.imread("image.jpg")


# random_scale_transform = A.RandomScale(
#     scale_limit=(-0.15, 0), interpolation=cv2.INTER_LINEAR)

# padding_transform = A.PadIfNeeded(
#     min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=0, always_apply=True)

# random_rotation_transform = A.Rotate(
#     limit=35, border_mode=0, value=0, always_apply=True)

# random_translate_transform = A.ShiftScaleRotate(
#     always_apply=True, border_mode=0, value=0, scale_limit=(-0.25, 0), rotate_limit=30)

iaa_affine_transform = A.Affine(
    shear=(-5, 5),
    scale=(0.75, 0.9), keep_ratio=True,
    rotate=(-30, 30),
    translate_percent=(0.0, 0.1),
    always_apply=True, cval=0)

brightness_transform = A.RandomBrightnessContrast(
    brightness_limit=0.05, contrast_limit=0.05, always_apply=True)

noise_transform = A.GaussNoise(var_limit=10.0, always_apply=True)

random_flip_transform = A.HorizontalFlip(p=0.5)


# Apply the transforms
transforms = A.Compose(
    [
        # random_scale_transform,
        # padding_transform,
        # random_rotation_transform,
        # random_translate_transform,
        iaa_affine_transform,
        random_flip_transform,
        brightness_transform,
        noise_transform,
    ])
data = {"image": image}
data_aug = transforms(**data)

# Get the augmented image
image_aug = data_aug["image"]

# Save the augmented image
cv2.imwrite("image_resized_padded.jpg", image_aug)
