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
  return dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_noisy_dir, train_clean_dir)
test_dataset = create_dataset(test_noisy_dir, test_clean_dir)
validation_dataset = create_dataset(validation_noisy_dir, validation_clean_dir)

# Define the Model 
class EAM(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    
    self.conv1 = Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')
    self.conv2 = Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu') 

    self.conv3 = Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')
    self.conv4 = Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')

    self.conv5 = Conv2D(64, (3,3),padding='same',activation='relu')

    self.conv6 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv7 = Conv2D(64, (3,3),padding='same')

    self.conv8 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv9 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv10 = Conv2D(64, (1,1),padding='same')

    self.gap = GlobalAveragePooling2D()

    self.conv11 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv12 = Conv2D(64, (3,3),padding='same',activation='sigmoid')

  def call(self,input):
    conv1 = self.conv1(input)
    conv1 = self.conv2(conv1)

    conv2 = self.conv3(input)
    conv2 = self.conv4(conv2)

    concat = concatenate([conv1,conv2])
    conv3 = self.conv5(concat)
    add1 = Add()([input,conv3])

    conv4 = self.conv6(add1)
    conv4 = self.conv7(conv4)
    add2 = Add()([conv4,add1])
    add2 = Activation('relu')(add2)

    conv5 = self.conv8(add2)
    conv5 = self.conv9(conv5)
    conv5 = self.conv10(conv5)
    add3 = Add()([add2,conv5])
    add3 = Activation('relu')(add3)

    gap = self.gap(add3)
    gap = Reshape((1,1,64))(gap)
    conv6 = self.conv11(gap)
    conv6 = self.conv12(conv6)
    
    mul = Multiply()([conv6, add3])
    out = Add()([input,mul]) # This is not included in the reference code
    return out
  
tf.keras.backend.clear_session()
input = Input(shape=(256, 256, 3))

conv1 = Conv2D(64, (3,3),padding='same')(input)

eam1 = EAM()(conv1)
eam2 = EAM()(eam1)
eam3 = EAM()(eam2)
eam4 = EAM()(eam3) 
conv2 = Conv2D(3, (3,3),padding='same')(eam4)

out = Add()([conv2,input])

RIDNet = Model(input,out)
RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError())

#traing for 20 epochs
RIDNet.fit(train_dataset, epochs=20, validation_data=test_dataset)  