import tensorflow as tf
import os
from tensorflow.keras.layers import (
  Conv2D,
  GlobalAveragePooling2D,
  concatenate,
  Add,
  Activation,
  Input,
  Reshape,
  Multiply,
  Cropping2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6048)])

        print("Physical GPU:", tf.config.list_physical_devices('GPU'))
    except RuntimeError as e:
        print(e)

def load_image_pair(noisy_dir, clean_dir):
  noisy_imgs = sorted(os.listdir(noisy_dir))
  clean_imgs = sorted(os.listdir(clean_dir))

  for n_img, c_img in zip(noisy_imgs, clean_imgs):
    n_img_path = os.path.join(noisy_dir, n_img)
    c_img_path = os.path.join(clean_dir, c_img)

    n_img = img_to_array(load_img(n_img_path, target_size=(256, 256))) / 255.0
    c_img = img_to_array(load_img(c_img_path, target_size=(256, 256))) / 255.0

    yield (n_img, c_img)

# Replace with your actual directories
train_noisy_dir = './datasets/train/input'
train_clean_dir = './datasets/train/groundtruth'
test_noisy_dir = './datasets/test/input'
test_clean_dir = './datasets/test/groundtruth'
validation_noisy_dir = './datasets/validation/input'
validation_clean_dir = './datasets/validation/groundtruth'

def create_dataset(noisy_dir, clean_dir):
  dataset = tf.data.Dataset.from_generator(
    load_image_pair,
    args=[noisy_dir, clean_dir],
    output_signature=(
      tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
      tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
    )
  )
  return dataset.batch(8).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_noisy_dir, train_clean_dir)
test_dataset = create_dataset(test_noisy_dir, test_clean_dir)
validation_dataset = create_dataset(validation_noisy_dir, validation_clean_dir)


# Load the model
model_name = 'models/RIDNet.h5'
RIDNet = tf.keras.models.load_model(model_name)

# Evaluate the model
result = RIDNet.evaluate(validation_dataset)
print("Validation Loss:", result)

RIDNet.save('RIDNet.h5')

# Load the model
RIDNet = tf.keras.models.load_model('RIDNet.h5')

# Evaluate the model
result = RIDNet.evaluate(validation_dataset)
print("Validation Loss:", result)

