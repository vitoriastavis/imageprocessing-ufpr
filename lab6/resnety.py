
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from pathlib import Path 
import shutil
import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


t_path = './resy/Treino/'
v_path = './resy/Valid/'


# loop over the input images
for folder in os.listdir(t_path):
    
    for filename in os.listdir(t_path+folder):

        image = cv.imread(t_path+folder+'/'+filename)
    
        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        cor = np.uint8([[[255, 217, 15]]])
        cor_hsv = cv.cvtColor(cor, cv.COLOR_RGB2HSV)
    
        # 25 240 255
        #yellow = (51, 94, 100)  #rgb(255, 217, 15)
        light = np.array([20,100,50])
        dark = np.array([40,255,255])
        mask = cv.inRange(hsv_img, light, dark)
       
        #target = cv.bitwise_and(hsv_img, hsv_img, mask = mask)  
        
        cv.imwrite(t_path+folder+'/'+filename, mask)
        
# loop over the input images
for folder in os.listdir(v_path):
    
    for filename in os.listdir(v_path+folder):

        image = cv.imread(v_path+folder+'/'+filename)
    
        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        cor = np.uint8([[[255, 217, 15]]])
        cor_hsv = cv.cvtColor(cor, cv.COLOR_RGB2HSV)
    
        # 25 240 255
        #yellow = (51, 94, 100)  #rgb(255, 217, 15)
        light = np.array([20,100,50])
        dark = np.array([40,255,255])
        mask = cv.inRange(hsv_img, light, dark)
       
        #target = cv.bitwise_and(hsv_img, hsv_img, mask = mask)  
        
        cv.imwrite(v_path+folder+'/'+filename, mask)
                

def crop_image(img, height=0, width=0):

    h = img.shape[0]
    h_crop = 0
    w = img.shape[1]
    w_crop = 0

    if height != 0:
        h_crop = int((height * h) / 100)
    else:
        h_crop = h

    if width != 0:
        w_crop = int((width * w) / 100)
    else:
        w_crop = w

    img = img[0:h_crop, 0:w_crop]

    return img


for folder in os.listdir(t_path):

    for filename in os.listdir(t_path+folder):
     
        img = cv.imread(t_path+folder+'/'+filename)

        cut = crop_image(img, 50, 0)

        cv.imwrite(t_path+folder+'/'+filename, cut)
    
for folder in os.listdir(v_path):

    for filename in os.listdir(v_path+folder):
     
        img = cv.imread(v_path+folder+'/'+filename)

        cut = crop_image(img, 50, 0)

        cv.imwrite(v_path+folder+'/'+filename, cut)


resnet50 = ResNet50()


row = 256
column = 256
input_shape = (row, column, 3)
batch_size = 32
#t_path = '/home/vitoria/home/processamento-de-imagens/lab6/Treino'
#v_path = '/home/vitoria/home/processamento-de-imagens/lab6/Valid'


#t_dataset = image_dataset_from_directory(t_path, labels = train_labels, label_mode ='categorical', image_size = (256,256), batch_size=batch_size, color_mode='rgb', shuffle=False)
#v_dataset = image_dataset_from_directory(v_path, labels = valid_labels, label_mode ='categorical', image_size = (256,256), batch_size=batch_size, color_mode='rgb', shuffle=False)


t_dataset = image_dataset_from_directory(t_path,
                                        image_size = (row,column),
                                        batch_size = batch_size, color_mode = 'rgb',
                                        shuffle = False)
v_dataset = image_dataset_from_directory(v_path,
                                        image_size = (row,column),
                                        batch_size = batch_size, color_mode = 'rgb',
                                        shuffle = False)
cnn = ResNet50(weights = 'imagenet', include_top = False,
                input_shape = input_shape)
inputs = keras.Input(shape = input_shape)
x = preprocess_input(inputs)
x = cnn(x)
output = GlobalAveragePooling2D()(x)
model = Model(inputs, output)

model.summary()

x_train = model.predict(t_dataset)
x_valid = model.predict(v_dataset)

y_train = np.concatenate([y for x, y in t_dataset], axis=0)
y_valid = np.concatenate([y for x, y in v_dataset], axis=0)

knn = KNeighborsClassifier(n_neighbors = 1, leaf_size = 1, n_jobs = -1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_valid)

#acc = model.score(v_dataset, y_valid)
acc = accuracy_score(y_valid, y_pred)
print()
print("------ Evaluating ResNet50 cut accuracy ------")
print()
print("ResNet50 cut accuracy: {:.2f}%".format(acc * 100))
print()
cm = confusion_matrix(y_valid, y_pred)
print (cm)
print()

print(classification_report(y_valid, y_pred))
print()
