import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

vgg16 = VGG16()

input_shape = (256, 256, 3)
batch_size = 32

t_path = './Treino/'
v_path = './Valid/'

train_dataset = image_dataset_from_directory(
        t_path,
        image_size=(256, 256),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False)

valid_dataset = image_dataset_from_directory(
        v_path,
        image_size=(256, 256),
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False)

cnn = VGG16(weights = 'imagenet', include_top = False,
            input_shape = (256, 256, 3))
inputs = keras.Input(shape = input_shape)
x = preprocess_input(inputs)
x = cnn(x)
output = Flatten()(x)
model = Model(inputs, output)
model.summary()

X_train = model.predict(train_dataset)
X_valid = model.predict(valid_dataset)

y_train = np.concatenate([y for x, y in train_dataset], axis=0)
y_valid = np.concatenate([y for x, y in valid_dataset], axis=0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)

print("------ Evaluating VGG accuracy ------")
print()
#acc = accuracy_score(y_valid, y_pred)
#print("VGG accuracy: {:.2f}%".format(acc * 100))
acc = knn.score(X_valid, y_valid)
print("VGG accuracy: {:.2f}%".format(acc * 100))
print()
print(confusion_matrix(y_valid, y_pred))
print()
print(classification_report(y_valid, y_pred))


print()
print("------ Evaluating VGG accuracy WITH GRIDSEARCH------")
print()
leaf_size = list(range(1,50))
#n_neighbors = list(range(1,30))
jobs = [1,2,4]
n_jobs = (list(jobs))
hyperparameters = dict(leaf_size = leaf_size, n_jobs = n_jobs)
knn_2 = KNeighborsClassifier()
grid = GridSearchCV(knn_2, hyperparameters, cv=10, scoring='accuracy',
                    return_train_score=False,verbose=1)

best_model_2 = grid.fit(x_train, y_train)

print()
print('Best 2 leaf_size:', best_model_2.best_estimator_.get_params()['leaf_size'])
print('Best 2 jobs:', best_model_2.best_estimator_.get_params()['n_jobs'])
print('Best 2 n_neighbors:', best_model_2.best_estimator_.get_params()['n_neighbors'])