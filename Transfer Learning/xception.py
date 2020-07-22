import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D ,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Input

from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.applications.xception import Xception

import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

train_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/train'
valid_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/valid'
test_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=20)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=4)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=10, shuffle=False)

input_tensor = Input(shape=(299, 299, 3))
base_model = Xception(input_tensor=input_tensor , weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=20,
          verbose=2
)
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
model.save('C.h5')
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

print('Confusion matrix, without normalization')

print(cm)
print((cm[0][0]+cm[1][1]+cm[2][2])/152.0)