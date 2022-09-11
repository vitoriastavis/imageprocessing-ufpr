from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import argparse
import cv2 as cv
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test_dataset", required=True,
	help="path to input test dataset")
ap.add_argument("-v", "--valid_dataset", required=True,
	help="path to input validation dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


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

def image_to_feature_vector(image, size = (32, 32)):
    	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	
	hist = cv.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 256, 0, 256, 0, 256])
	
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 
	else:
		cv.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

# grab the list of images that we'll be describing
test_images = list(paths.list_images(args["test_dataset"]))
valid_images = list(paths.list_images(args["valid_dataset"]))

# initialize the raw pixel intensities matrix, 
# the features matrix, and labels list
t_rawImages = []
t_features = []
t_labels = []
v_rawImages = []
v_features = []
v_labels = []


for filename in test_images:

    print(filename)   
    img = cv.imread(filename)
    cut = crop_image(img, 50, 0)

    cv.imwrite(filename, cut)

for filename in valid_images:

 #   print(filename)
    img = cv.imread(filename)
    cut = crop_image(img, 50, 0)

    cv.imwrite(filename, cut)

    #for filename in os.listdir(t_path+folder):

     #   print(filename)
      #  print(t_path+folder+'/'+filename)
       # img = cv.imread(t_path+folder+'/'+filename)

        #cut = crop_image(img, 50, 0)

        #cv.imwrite(t_path+folder+'/'+filename, cut)

#i = 0
# loop over the input images
#for filename in test_images:

 #   image = cv.imread(filename)
  #  hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   # cor = np.uint8([[[255, 217, 15]]])
    #cor_hsv = cv.cvtColor(cor, cv.COLOR_RGB2HSV)
   
  
    #light = np.array([20,100,50])
    #dark = np.array([40,255,255])
    #mask = cv.inRange(hsv_img, light, dark)
    #nome = str(i)
    #target = cv.bitwise_and(hsv_img, hsv_img, mask = mask)  
    
    #plt.imsave(filename, mask)    
    
    #i = i + 1


# loop over the input images
#for filename in valid_images:

 #   image = cv.imread(filename)
   
  #  hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   # cor = np.uint8([[[255, 217, 15]]])
    #cor_hsv = cv.cvtColor(cor, cv.COLOR_RGB2HSV)
   
    # 25 240 255
    #yellow = (51, 94, 100)  #rgb(255, 217, 15)
    #light = np.array([20,100,50])
    #dark = np.array([40,255,255])
    #mask = cv.inRange(hsv_img, light, dark)
    #nome = str(i)
    #target = cv.bitwise_and(hsv_img, hsv_img, mask = mask)  
    
  #  plt.imsave(filename, mask)
    
   # i = i + 1

# loop over the input images
for (i, test_images) in enumerate(test_images):
    
	# load the image and extract the class label  
	image = cv.imread(test_images)
	label = test_images.split(os.path.sep)[-1].split(".")[0]
	label = label[:-3]
 
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	t_rawImages.append(pixels)
	t_features.append(hist)
	t_labels.append(label)
 

# loop over the input images
for (i, valid_images) in enumerate(valid_images):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv.imread(valid_images)
	label = valid_images.split(os.path.sep)[-1].split(".")[0]
	label = label[:-3]
 
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	v_rawImages.append(pixels)
	v_features.append(hist)
	v_labels.append(label)
  
t_rawImages = np.array(t_rawImages)
t_features = np.array(t_features)
t_labels = np.array(t_labels)

v_rawImages = np.array(v_rawImages)
v_features = np.array(v_features)
v_labels = np.array(v_labels)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("------ Evaluating raw pixel accuracy ------ ")
print()
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(t_rawImages, t_labels)
acc = model.score(v_rawImages, v_labels)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))
print()
y_pred = model.predict(v_rawImages)
cm = confusion_matrix(v_labels, y_pred)
print (cm)
print()
print(classification_report(v_labels, y_pred))
print()

# train and evaluate a k-NN classifer on the histogram
print("------ Evaluating histogram accuracy ------")
print()
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(t_features, t_labels)
acc = model.score(v_features, v_labels)
print("Histogram accuracy: {:.2f}%".format(acc * 100))
print()
y_pred = model.predict(v_features)
cm = confusion_matrix(v_labels, y_pred)
print (cm)
print()
print(classification_report(v_labels, y_pred))
print()