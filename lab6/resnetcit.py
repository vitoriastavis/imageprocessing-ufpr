
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


t_path = './cur/Treino/'
v_path = './cur/Valid/'


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


#for folder in os.listdir(t_path):

   # print(folder)

    #for filename in os.listdir(t_path+folder):

     #   print(filename)
      #  print(t_path+folder+'/'+filename)
       # img = cv.imread(t_path+folder+'/'+filename)

        #cut = crop_image(img, 50, 0)

        #cv.imwrite(t_path+folder+'/'+filename, cut)


resnet50 = ResNet50()


row = 256
column = 256
input_shape = (row, column, 3)
batch_size = 256

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

knn = KNeighborsClassifier(n_neighbors = 1)
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