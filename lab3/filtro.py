# Lab 03 - Remoção de ruído com filtros
# filtro.py - remove o ruído com o método especificado
# Vitória Stavis de Araujo - GRR20200243

import sys
import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt

import tests

def main(argv):
    
    # ler a imagem
    img = cv.imread(argv[1], 0)   
    level = float(argv[2])    
    method = int(argv[3])
    
    noise_img = tests.sp_noise(img, level)
   
    # remove o ruído de acordo com o método
    if method == 0:       
        if level <= 0.02:               #média com gaussian blur
            res = tests.average_gaussian(noise_img, 3)
        else:    
            res = tests.average_gaussian(noise_img, 7)
            
    elif method == 1:                   #mediana
        res = tests.median(noise_img, 3)
        
    elif method == 2:                   #stacking   
        res = tests.stacking(img, 150, level)
        
    elif method == 3:                   # média com filter2d
        res = tests.average_2d(noise_img, 17) 
        
    elif method == 4:                   # média com blur                                       
        if level <= 0.02:
            res = tests.average_blur(noise_img, 2) 
        else:
            res = tests.average_blur(noise_img, 4) 
      
    # salva a imagem resultante
    #plt.imsave(argv[4], res)
    cv.imwrite(argv[4], res)
    
if __name__ == '__main__':
    main(sys.argv)
    
    