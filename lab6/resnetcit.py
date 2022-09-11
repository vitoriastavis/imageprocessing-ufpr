
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


for folder in os.listdir(t_path):

    print(folder)

    for filename in os.listdir(t_path+folder):

        print(filename)
        print(t_path+folder+'/'+filename)
        img = cv.imread(t_path+folder+'/'+filename)

        cut = crop_image(img, 50, 0)

        cv.imwrite(t_path+folder+'/'+filename, cut)