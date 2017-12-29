#! /usr/bin/env python


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#from keras import backend as K
#K.set_image_dim_ordering('th')
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#from keras.optimizers import SGD
from keras import optimizers
from keras.layers import Conv2D
from keras import applications
import numpy as np
import sys
import argparse
from keras.utils import to_categorical

_train_dir = '/path/to/training/dir'
_val_dir = '/path/to/validation/dir'

## Part 1: Building CNN
model = Sequential()

# 1.Convolution layer
model.add(Convolution2D(_mask,3,3, input_shape = (128,128,3), activation = 'relu', border_mode = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
# 2.Convolution layer
model.add(Convolution2D(_mask,3,3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

# 2.Convolution layer
model.add(Convolution2D(_mask,3,3,  activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

# Flatten
model.add(Flatten())

# Full connection
model.add(Dense(output_dim = 128, activation = 'relu'))

 ## Part 2: Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(_train_dir,
                                                    target_size = (128,128),
                                                    batch_size = 32,
                                                    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(_val_dir,
                                                        target_size=(128,128),
                                                        batch_size = 32,
                                                        class_mode='categorical')




model.add(Dense(output_dim = len(train_generator.class_indices), activation = 'softmax'))

model.summary()
# Compiling CNN
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=289,epochs=5,
                        validation_data=validation_generator,validation_steps=58)


'''
## Part 3: Making new predictions
from keras.preprocessing import image

test_image = image.load_img(_test_img, target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)

result = model.predict(test_image)
#print train_datagen.class_indices


if result[0][0] == 1:
    predictions = 'EarthQuake'
else:
    predictions = 'Others'

print predictions
'''

