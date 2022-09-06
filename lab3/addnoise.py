import numpy as np
import random
import cv2

def sp_noise(image,prob):
        
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('img.png',0) 

noises = (0.01, 0.02, 0.05, 0.07, 0.1)
for i in noises:
    noise_img = sp_noise(image,i)
    cv2.imwrite('img-noise'+i+'.png', noise_img)


