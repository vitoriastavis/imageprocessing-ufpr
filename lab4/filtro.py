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
    method = int(argv[2])    
      
    # remove o ruído de acordo com o método
    if method == 0:          
        res = tests.average_gaussian(img, 7)
            
    elif method == 1:                   #mediana
        res = tests.median(img, 3)
        
    elif method == 2:                   #stacking   
        res = tests.stacking(img, 150, 0.7)
        
    elif method == 3:                   # média com filter2d
        res = tests.average_2d(img, 17) 
        
    elif method == 4:                  # média com blur                                       
        res = tests.average_blur(img, 4) 
      
    # salva a imagem resultante
    plt.imsave(argv[3], res)
    
if __name__ == '__main__':
    main(sys.argv)
    
    