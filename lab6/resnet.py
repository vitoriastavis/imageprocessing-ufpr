import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_imput
from pathlib import Path 
import shutil

resnet50 = ResNet50()

path_train = '/Treino'
path_valid = '/Valid'

train_dataset = image_dataset_from_directory(path_train, )