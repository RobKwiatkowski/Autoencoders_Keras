#!/usr/bin/env python
# coding: utf-8

#%% Reading Libraries

__author__ = "Robert Kwiatkowski"
__license__ = "GPL"

import warnings
warnings.filterwarnings("ignore")

# Importing keras components
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import LeakyReLU
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_normal

from sklearn.model_selection import train_test_split  # module for splitting data into train and test sets
from skimage.color import rgb2lab, lab2rgb  # module for converting RGB color model to LAB and back
from skimage.transform import resize
from skimage.io import imsave
from tqdm import tqdm  # module for displaying a progress bar

import numpy as np
import matplotlib.pyplot as plt  # library for visualisations
import tensorflow as tf  # library with NN core components

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
session = tf.compat.v1.InteractiveSession(config=config)

from tensorflow.python.client import device_lib  # module for checking available local devices

print(tf.__version__)
print(device_lib.list_local_devices())

%% Reading Data
path_imgs = r"\Inputs"

img_size=(256,256)

#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)

#Resize images, if needed
train = train_datagen.flow_from_directory(path_imgs, 
                                          target_size=img_size, 
                                          batch_size=340, 
                                          class_mode=None)



#%% Convert from RGB to Lab

X =[]
Y =[]
for img in tqdm(train[0]):
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128)
  except:
     print('error')
     
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
samples = len(X)
del X, Y

#%% Construction phase
np.random.seed(124)
leaky_alpha = 0.1

#Encoder

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), strides=2, input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal(), strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal(), strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal(), strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(1024, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

#Decoder

model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))

model.add(Conv2D(32, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(16, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(8, (3,3), padding='same', kernel_initializer=glorot_normal()))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=leaky_alpha))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', kernel_initializer=glorot_normal()))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


#%% Execution phase
batch = 28
epochs = 400
path_model = r"models\beach.model"

es = EarlyStopping(monitor='val_loss', 
                   min_delta=0.0001, 
                   patience=75, 
                   verbose=0, 
                   mode='auto', 
                   baseline=None, 
                   restore_best_weights=True)

checkpt = ModelCheckpoint(path_model, 
                          monitor='val_loss',
                          verbose=0,
                          save_best_only=True,
                          mode='auto')

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val), 
                    epochs=epochs,
                    batch_size=batch, 
                    callbacks=[es, checkpt])
model.save(path_model)


#%% plot training history 1
plt.plot(history.history['loss'][20:], label='train')
plt.plot(history.history['val_loss'][20:], label='test')
plt.title("Lossess")
plt.legend()
plt.show()


#%% plot training history 2
plt.plot(history.history['accuracy'][10:], label='train')
plt.plot(history.history['val_accuracy'][10:], label='test')
plt.legend()
plt.show()

img1_color = []
img1 = img_to_array(load_img(r'validation/inputs/black.jpg'))
img1 = resize(img1 ,(256,256))
img1_color.append(img1)
img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))

output1 = model.predict(img1_color)
output1 = output1*128
result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]



