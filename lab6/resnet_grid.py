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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

# current directory
directory = os.getcwd()  


resnet50 = ResNet50()

input_shape = (256, 256, 3)
batch_size = 32
#t_path = '/home/vitoria/home/processamento-de-imagens/lab6/Treino'
#v_path = '/home/vitoria/home/processamento-de-imagens/lab6/Valid'

t_path = './Treino/'
v_path = './Valid/'


#t_dataset = image_dataset_from_directory(t_path, labels = train_labels, label_mode ='categorical', image_size = (256,256), batch_size=batch_size, color_mode='rgb', shuffle=False)
#v_dataset = image_dataset_from_directory(v_path, labels = valid_labels, label_mode ='categorical', image_size = (256,256), batch_size=batch_size, color_mode='rgb', shuffle=False)


t_dataset = image_dataset_from_directory(t_path,
                                        image_size = (256,256),
                                        batch_size = 32, color_mode = 'rgb',
                                        shuffle = False)
v_dataset = image_dataset_from_directory(v_path,
                                        image_size = (256,256),
                                        batch_size = 32, color_mode = 'rgb',
                                        shuffle = False)

cnn = ResNet50(weights = 'imagenet', include_top = False,
                input_shape = (256, 256, 3))
inputs = keras.Input(shape = (256, 256, 3))
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
print("------ Evaluating ResNet50 accuracy ------")
print()
print("ResNet50 accuracy: {:.2f}%".format(acc * 100))
print()
cm = confusion_matrix(y_valid, y_pred)
print (cm)
print()

print(classification_report(y_valid, y_pred))
print()


print()
print("------ Evaluating ResNet50 accuracy WITH GRIDSEARCH------")
print()

leaf_size = list(range(1,50))
#n_neighbors = list(range(1,30))
jobs = [1,2,4]
n_jobs = (list(jobs))
hyperparameters = dict(leaf_size = leaf_size, n_jobs = n_jobs)
knn_2 = KNeighborsClassifier()

grid = GridSearchCV(knn_2, hyperparameters, cv=10, scoring='accuracy',
                    return_train_score=False,verbose=1)            

#Fit the model
best_model_2 = grid.fit(x_train, y_train)

#Print The value of best Hyperparameters
#print('Best 1 leaf_size:', best_model_1.best_estimator_.get_params()['leaf_size'])
#print('Best 1 jobs:', best_model_1.best_estimator_.get_params()['n_jobs'])
#print('Best 1 n_neighbors:', best_model_1.best_estimator_.get_params()['n_neighbors'])
print()
print('Best 2 leaf_size:', best_model_2.best_estimator_.get_params()['leaf_size'])
print('Best 2 jobs:', best_model_2.best_estimator_.get_params()['n_jobs'])
print('Best 2 n_neighbors:', best_model_2.best_estimator_.get_params()['n_neighbors'])

