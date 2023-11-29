import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
import shutil

E = -2.0 #-4.0 to 4.0
# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.BlendAlphaFrequencyNoise(
        exponent=E,
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=False),
        size_px_max=32,
        upscale_method="linear",
        iterations=1,
        sigmoid=False
    ),
    iaa.AdditiveGaussianNoise(scale=0.099 * 255),
    iaa.Salt(p=0.01)
])

# Set the path to the dataset directory
dataset_dir = "./dataset128"
train_dir = "./datasets/train/"
test_dir = "./datasets/test/"
validate_dir = "./datasets/validate/"

# prepare the directories
os.makedirs(train_dir + "input", exist_ok=True)
os.makedirs(train_dir + "groundtruth", exist_ok=True)
os.makedirs(test_dir + "input", exist_ok=True)
os.makedirs(test_dir + "groundtruth", exist_ok=True)
os.makedirs(validate_dir + "input", exist_ok=True)
os.makedirs(validate_dir + "groundtruth", exist_ok=True)    





# Get the list of image filenames
image_filenames = [filename for filename in os.listdir(dataset_dir) if filename.endswith(".jpg") or filename.endswith(".png")]

# Shuffle the image filenames
random.shuffle(image_filenames)

# Calculate the number of images for each set
num_images = len(image_filenames)
num_train = int(num_images * 0.8)
num_test = int(num_images * 0.15)
num_vali = num_images - num_train - num_test


# Iterate over the image filenames
for i, filename in enumerate(image_filenames):
    # Read the image
    image = cv2.imread(os.path.join(dataset_dir, filename))

    # Augment the image
    augmented_image = seq.augment_image(image)

    # Determine the destination directory based on the proportion
    if i < num_train:
        dest_dir = train_dir
    elif i < num_train + num_test:
        dest_dir = test_dir
    else:
        dest_dir = validate_dir

    # Save the original image to the groundtruth directory
    shutil.copy(os.path.join(dataset_dir, filename), os.path.join(dest_dir, "groundtruth", filename))

    # Save the augmented image to the input directory
    cv2.imwrite(os.path.join(dest_dir, "input", filename), augmented_image)

    # Print progress
    print(f"Processed {i+1}/{len(image_filenames)} images")

