import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from math import *
from PIL import Image
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from tensorflow import keras
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import itertools
import shutil
import random

train_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/train'
valid_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/valid'
test_path = '/Users/anubhavsharma/PycharmProjects/neuralnetwork1/cdata/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=20)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=4)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['covid', 'normal' , 'pneumonia'], batch_size=10, shuffle=False)

xtrain_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=20)
xvalid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=4)
xtest_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(299,299), classes=['covid', 'normal' , 'pneumonia'], batch_size=10, shuffle=False)


v16 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/v16.h5')
v19 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/v19.h5')
mnV2 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/mnv2.h5')
rn50  = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/rn50.h5')
rn50  = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/rn50.h5')
xcpn  = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/model/xcpn.h5')


p1 = v19.predict(x=test_batches, steps=len(test_batches), verbose=0)
p2 = v16.predict(x=test_batches, steps=len(test_batches), verbose=0)
p3 = mnV2.predict(x=test_batches, steps=len(test_batches), verbose=0)
p4 = rn50.predict(x=test_batches, steps=len(test_batches), verbose=0)
p5 = xcpn.predict(x=xtest_batches, steps=len(xtest_batches), verbose=0)


cm1: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(p1, axis=-1))
print(cm1)
print("p1: " , (cm1[0][0]+cm1[1][1]+cm1[2][2])/1253.0)
cm2: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(p2, axis=-1))
print(cm2)
print("p2: " ,(cm2[0][0]+cm2[1][1]+cm2[2][2])/1253.0)
cm3: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(p3, axis=-1))
print(cm3)
print("p3: " ,(cm3[0][0]+cm3[1][1]+cm3[2][2])/1253.0)
cm4: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(p4, axis=-1))
print(cm4)
print("p4: " ,(cm4[0][0]+cm4[1][1]+cm4[2][2])/1253.0)
cm5: object = confusion_matrix(y_true=xtest_batches.classes, y_pred=np.argmax(p5, axis=-1))
print(cm5)
print("p5: " ,(cm5[0][0]+cm5[1][1]+cm5[2][2])/1253.0)




pfinal6 = []
for (a,b) in zip(p1,p4):
    pfinal6.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal6, axis=-1))
print(cm)
print("pf6: p1 p4: ",(cm[0][0]+cm[1][1]+cm[2][2])/1253.0)

pfinal1 = []
for (a,b) in zip(p1,p2):
    pfinal1.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal1, axis=-1))
print(cm)
print("pf1: p1 p2: ",(cm[0][0]+cm[1][1]+cm[2][2])/1253.0)

pfinal3 = []
for (a,b) in zip(p2,p3):
    pfinal3.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal3, axis=-1))
print(cm)
print("pf3: p3 p2: ",(cm[0][0]+cm[1][1]+cm[2][2])/1253.0)

pfinal4 = []
for (a,b) in zip(p3,p4):
    pfinal4.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal4, axis=-1))
print(cm)
print("pf4: p3 p4: ",(cm[0][0]+cm[1][1]+cm[2][2])/1253.0)

pfinal5 = []
for (a,b) in zip(p1,p5):
    pfinal5.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal5, axis=-1))
print(cm)
print("pf5: p1 p5: ",(cm[0][0]+cm[1][1]+cm[2][2])/1253.0)




'''
pfinal9 = []
for (a,b) in zip(pfinal1,pfinal3):
    pfinal9.append([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0 , (a[2]+b[2])/2.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal9, axis=-1))
print(cm)
print("pf1 pf3: ",(cm[0][0]+cm[1][1]+cm[2][2])/152.0)



pfinal7 = []
for (a,b,c) in zip(pfinal6,pfinal5,pfinal):
    pfinal7.append([(a[0]+b[0]+c[0])/3.0, (a[1]+b[1]+c[1])/3.0 , (a[2]+b[2]+c[2])/3.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal7, axis=-1))
print(cm)
print("p2 p3 p4: ",(cm[0][0]+cm[1][1]+cm[2][2])/152.0)

pfinal8 = []
for (a,b,c,d) in zip(p1,p2,p3,p4):
    pfinal8.append([(a[0]+b[0]+c[0]+d[0])/4.0, (a[1]+b[1]+c[1]+d[1])/4.0 , (a[2]+b[2]+c[2]+d[2])/4.0 ])
#print(pfinal)
cm: object = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pfinal8, axis=-1))
print(cm)
print("1234: ",(cm[0][0]+cm[1][1]+cm[2][2])/152.0)

'''
