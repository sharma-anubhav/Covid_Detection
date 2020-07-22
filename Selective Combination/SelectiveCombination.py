import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D ,GlobalAveragePooling2D , Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from keras.initializers import RandomNormal
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob

import warnings

train_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/train'
valid_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/valid'
test_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=20)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=4)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=10, shuffle=False)

v19 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/v19.h5')
mnV2 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/mnv2.h5')
rn50  = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/rn50.h5')

p1 = v19.predict(x=test_batches, steps=len(test_batches), verbose=0)
p3 = mnV2.predict(x=test_batches, steps=len(test_batches), verbose=0)
p4 = rn50.predict(x=test_batches, steps=len(test_batches), verbose=0)

pEnsamble = []
for (a,b) in zip(p3,p4):
    pEnsamble.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])

p2 = np.array(pEnsamble).reshape(1253,3)
p5 = p1
p1_class = np.argmax(p1, axis=-1)
p2_class = np.argmax(p2, axis=-1)

pFinal = []

for (a,b) in zip(p1_class,p2_class):
    pFinal.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])

cm: object = confusion_matrix(y_true=test_batches.classes, y_pred= np.argmax(pFinal, axis=-1))

print('Statistical')
print(cm)
print((cm[0][0]+cm[1][1]+cm[2][2])/1253.0)
