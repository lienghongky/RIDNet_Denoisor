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
from ModelClass import EAM, RIDNetModel

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
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
# datasets directory structure
# datasets
# ├── train
# │   ├── groundtruth
# │   │   ├── 0001.png
# │   │   ├── 0002.png
# │   │   ├── ...
# │   │   └── 1000.png
# │   └── input
# │       ├── 0001.png
# │       ├── 0002.png
# │       ├── ...
# │       └── 1000.png
# ├── test
# │   ├── groundtruth
# │   │   ├── 0001.png
# │   │   ├── 0002.png
# │   │   ├── ...
# │   │   └── 100.png
# │   └── input
# │       ├── 0001.png
# │       ├── 0002.png
# │       ├── ...
# │       └── 100.png
# └── validation
#     ├── groundtruth
#     │   ├── 0001.png
#     │   ├── 0002.png
#     │   ├── ...
#     │   └── 100.png
#     └── input
#         ├── 0001.png
#         ├── 0002.png
#         ├── ...
#         └── 100.png

# Replace with your actual directories
train_noisy_dir = './datasets/train/input'
train_clean_dir = './datasets/train/groundtruth'
# train_noisy_dir = './datasets/lol/input'
# train_clean_dir = './datasets/lol/groundtruth'
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
  return dataset.batch(1).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_noisy_dir, train_clean_dir)
test_dataset = create_dataset(test_noisy_dir, test_clean_dir)
validation_dataset = create_dataset(validation_noisy_dir, validation_clean_dir)


RIDNet = RIDNetModel # Model(input,out)
RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError())


# Define the checkpoint path
checkpoint_path = 'model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.keras'

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_freq='epoch')

model_name = 'models/LATES_MODEL.keras'
RIDNet = tf.keras.models.load_model(model_name,custom_objects={'EAM':EAM})

# Train the model with the checkpoint callback
RIDNet.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[checkpoint_callback])

# Save the final model

model_name = f'LATES_MODEL.keras'
RIDNet.save(model_name)

# Evaluate the model
result = RIDNet.evaluate(validation_dataset)
print("Validation Loss:", result)

