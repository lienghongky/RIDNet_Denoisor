import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
import shutil



# #copy all filss from train-scene to dataset128 with new name
# def copy_files(src_dir, dst_dir, prefix="ts-"):
#     # Get the list of image filenames in the source directory
#     image_filenames = [filename for filename in os.listdir(src_dir) if filename.endswith((".png", ".jpg", ".jpeg"))]

#     # Initialize counters
#     count_copied = 0
#     total_files = len(image_filenames)

#     # Iterate over the image filenames
#     for filename in image_filenames:
#         src = os.path.join(src_dir, filename)
#         dst_filename = os.path.basename(filename)
#         dst = os.path.join(dst_dir, f"{prefix}-{dst_filename}")
#         shutil.copyfile(src, dst)
#         count_copied += 1
#         print(f"{count_copied}/{total_files} Copy {src} to {dst}")

# src_dirs = [
#             "unziped_datasets/dataset_phphy",
#             # "unziped_datasets/dataset_ffhq", 
#             # "unziped_datasets/dataset_scene",
#             # "unziped_datasets/dataset_ssid", 
#             # "unziped_datasets/dataset_urban"
#             ] 
# dst_dir = "./dataset_all"
# for src_dir in src_dirs:
#     copy_files(src_dir, dst_dir, prefix=src_dir[25:])
# breakpoint()



E = -2.0 #-4.0 to 4.0
# Define the augmentation sequence
seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Sequential([
                                iaa.BlendAlphaFrequencyNoise(
                                    exponent=E,
                                    foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
                                    size_px_max=32,
                                    upscale_method="nearest",
                                    iterations=1,
                                    sigmoid=False
                                    ),
                                iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                                iaa.SaltAndPepper(p=0.2)
                            ]),
                iaa.Sequential([
                                iaa.JpegCompression(compression=(20, 34)),
                                iaa.OneOf([
                                    iaa.AdditiveGaussianNoise(scale=(0.1, 0.3 * 255), per_channel=True),
                                    iaa.AdditivePoissonNoise(5),
                                ]),
                                iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True),
                                iaa.OneOf([
                                    iaa.SaltAndPepper(p=0.2),
                                    iaa.Salt(p=0.1),
                                    iaa.Pepper(p=0.1),
                                    ]),
                                iaa.BlendAlphaFrequencyNoise(
                                    exponent=E,
                                    foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
                                    size_px_max=32,
                                    upscale_method="nearest",
                                    iterations=1,
                                    sigmoid=False
                                    ),
                            ])
            ])
])


# Set the path to the dataset directory
dataset_dir = "./dataset_all/"
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
num_train = int(num_images * 0.90)
num_test = int(num_images * 0.05)
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
    file_ext = filename.split(".")[-1]
    outputFileName = f"d100k_grouped-{i}.{file_ext}"
    # Save the original image to the groundtruth directory
    shutil.copy(os.path.join(dataset_dir, filename), os.path.join(dest_dir, "groundtruth", outputFileName))

    # Save the augmented image to the input directory
    cv2.imwrite(os.path.join(dest_dir, "input", outputFileName), augmented_image)

    # Print progress
    print(f"Processed {i+1}/{len(image_filenames)} -> {outputFileName}")

