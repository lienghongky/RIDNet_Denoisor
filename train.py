import tensorflow as tf
import tensorflow as tf
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

# Define the EAM layer
class EAM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(64, (3, 3), dilation_rate=1, padding='same', activation='relu')
        self.conv2 = Conv2D(64, (3, 3), dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv2D(64, (3, 3), dilation_rate=3, padding='same', activation='relu')
        self.conv4 = Conv2D(64, (3, 3), dilation_rate=4, padding='same', activation='relu')
        self.conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv6 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv7 = Conv2D(64, (3, 3), padding='same')
        self.conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv10 = Conv2D(64, (1, 1), padding='same')
        self.gap = GlobalAveragePooling2D()
        self.conv11 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv12 = Conv2D(64, (3, 3), padding='same', activation='sigmoid')

    def call(self, input):
        conv1 = self.conv1(input)
        conv1 = self.conv2(conv1)
        conv2 = self.conv3(input)
        conv2 = self.conv4(conv2)
        concat = concatenate([conv1, conv2])
        conv3 = self.conv5(concat)
        add1 = Add()([input, conv3])
        conv4 = self.conv6(add1)
        conv4 = self.conv7(conv4)
        add2 = Add()([conv4, add1])
        add2 = Activation('relu')(add2)
        conv5 = self.conv8(add2)
        conv5 = self.conv9(conv5)
        conv5 = self.conv10(conv5)
        add3 = Add()([add2, conv5])
        add3 = Activation('relu')(add3)
        gap = self.gap(add3)
        gap = Reshape((1, 1, 64))(gap)
        conv6 = self.conv11(gap)
        conv6 = self.conv12(conv6)
        mul = Multiply()([conv6, add3])
        out = Add()([input, mul])
        return out

# Clear previous session
tf.keras.backend.clear_session()

# Define the input layer
input = Input(shape=(128, 128, 3))
padded_input = tf.keras.layers.ZeroPadding2D(padding=((64, 64), (64, 64)))(input)

# Apply convolutional layers
conv1 = Conv2D(64, (3, 3), padding='same')(padded_input)

# Apply EAM layers
eam1 = EAM()(conv1)
eam2 = EAM()(eam1)
eam3 = EAM()(eam2)
eam4 = EAM()(eam3)

# Apply final convolutional layer
conv2 = Conv2D(3, (3, 3), padding='same')(eam4)

# Crop the output back to the original size
cropped_output = Cropping2D(cropping=((64, 64), (64, 64)))(conv2)

# Define the model
model = Model(inputs=input, outputs=cropped_output)
from tensorflow.keras.losses import MeanSquaredError

# dataset preparing and loading
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './datasets/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './datasets/test',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32)









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
input = Input(shape=(128, 128, 3))
padded_input = tf.keras.layers.ZeroPadding2D(padding=((64, 64), (64, 64)))(input)

conv1 = Conv2D(64, (3,3),padding='same')(padded_input)

eam1 = EAM()(conv1)
eam2 = EAM()(eam1)
eam3 = EAM()(eam2)
eam4 = EAM()(eam3) 
conv2 = Conv2D(3, (3,3),padding='same')(eam4)

# Crop the output back to the original size
cropped_output = Cropping2D(cropping=((64, 64), (64, 64)))(conv2)

out = Add()([cropped_output,input])

RIDNet = Model(input,out)
RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError())

#traing for 20 epochs
RIDNet.fit(train_dataset, epochs=20, validation_data=test_dataset)  