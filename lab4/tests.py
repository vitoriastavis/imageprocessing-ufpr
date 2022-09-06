# Lab 03 - Remoção de ruído com filtros
# tests.py - experimentação com filtros diferentes
# Vitória Stavis de Araujo - GRR20200243

# importar modulos necessarios
from curses import reset_prog_mode
from pickletools import uint8
import sys
import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt

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


## FUNCOES AVERAGE FILTER 2D
def average_2d(img, size):
    
    kernel = np.ones((size, size), np.float32)/255
    res = cv.filter2D(img, -1, kernel)
    
    return res  
   
def test_av_2d(img, noises):
    
    print('----- Average Filter 2D tests -----')
     
    for i in noises:
        noise = sp_noise(img, i)  
  
        psnr = cv.PSNR(img, average_2d(noise, 15))
        print('Noise: ', i, 'Average 2D 15, PSNR: ', round(psnr, 3))
        
        psnr = cv.PSNR(img, average_2d(noise, 17))
        print('Noise: ', i, 'Average 2D 17, PSNR: ', round(psnr, 3))
        
        psnr = cv.PSNR(img, average_2d(noise, 19))
        print('Noise: ', i, 'Average 2D 19, PSNR: ', round(psnr, 3))
        
        print('\n')
        

## FUNCOES AVERAGE BLUR    
def average_blur(img, size):
    
    res = cv.blur(img, ksize = (size,size))   
    return res   

def test_av_blur(img, noises):
    
    print('----- Average Blur tests -----')
    
    for i in noises:
        
        noise = sp_noise(img, i) 
        
        psnr = cv.PSNR(img, average_blur(noise, 2))
        print('Noise: ', i, 'Average blur 2, PSNR: ', round(psnr, 3))
        
        psnr = cv.PSNR(img, average_blur(noise, 4))
        print('Noise: ', i, 'Average blur 4, PSNR: ', round(psnr, 3))
        
        psnr = cv.PSNR(img, average_blur(noise, 6))
        print('Noise: ', i, 'Average blur 6, PSNR: ', round(psnr, 3))
        
        print('\n') 
        
        
## FUNCOES AVERAGE GAUSSIAN BLUR     
def average_gaussian(img, size):    
    
    res = cv.GaussianBlur(src = img, ksize =(size, size), sigmaX = 0)
    return res   
        
def test_av_gauss(img, noises):
    
    print('----- Average Gaussian Blur tests -----')
    
    for i in noises:
        noise = sp_noise(img, i) 
        
        psnr = cv.PSNR(img, average_gaussian(noise, 3))
        print('Noise: ', i, 'Average gaussian 3, PSNR: ', round(psnr, 3))
            
        psnr = cv.PSNR(img, average_gaussian(noise, 5))
        print('Noise: ', i, 'Average gaussian 5, PSNR: ', round(psnr, 3))
            
        psnr = cv.PSNR(img, average_gaussian(noise, 7))
        print('Noise: ', i, 'Average gaussian 7, PSNR: ', round(psnr, 3))
            
        print('\n') 
        


## FUNCOES MEDIAN 
def median(img, size):
    
    res = cv.medianBlur(img, size)
    return res

def test_median(img, noises):
    
    print('----- Median filter tests -----')
    
    for i in noises:
        noise = sp_noise(img, i) 
        
        psnr = cv.PSNR(img, median(noise, 3))
        print('Noise: ', i, 'Median 3, PSNR: ', round(psnr, 3))
            
        psnr = cv.PSNR(img, median(noise, 5))
        print('Noise: ', i, 'Median 5, PSNR: ', round(psnr, 3))
            
        psnr = cv.PSNR(img, median(noise, 7))
        print('Noise: ', i, 'Median 7, PSNR: ', round(psnr, 3))            
            
        print('\n') 
        
        
## FUNCOES STACKING           
def stacking(img, n, level):
    
    # array do tipo uint32 para não dar overflow
    stacked = np.zeros(img.shape, np.uint32)      
    
    for i in range(n):
        # cria imagem com ruído e soma
        noise = sp_noise(img, level)
        stacked += noise
    
    # divide pelo total de imagens pra dar a média
    stacked //= n
    
    res = stacked.astype(np.uint8)
    
    return res
        
def test_stacking(img, noises):
    
    print('----- Image stacking tests -----')
    
    for i in noises:                    
    
        res = stacking(img, 15, i)    
        psnr = cv.PSNR(img, res)
        print('Noise: ', i, 'Stacking 15, PSNR: ', round(psnr, 3))
        
        res = stacking(img, 25, i)
        psnr = cv.PSNR(img, res)
        print('Noise: ', i, 'Stacking 25, PSNR: ', round(psnr, 3)) 
        
        res = stacking(img, 50, i)        
        psnr = cv.PSNR(img, res)
        print('Noise: ', i, 'Stacking 50, PSNR: ', round(psnr, 3)) 
        
        res = stacking(img, 100, i) 
        psnr = cv.PSNR(img, res)
        print('Noise: ', i, 'Stacking 100, PSNR: ', round(psnr, 3)) 
        
        res = stacking(img, 150, i) 
        psnr = cv.PSNR(img, res)
        print('Noise: ', i, 'Stacking 150, PSNR: ', round(psnr, 3))  
        
        print('\n') 


def main(argv):
    
    # ler a imagem
    img = cv.imread(argv[1], 0)         

    noises = (0.01, 0.02, 0.05, 0.07, 0.1) 
        
    test_av_2d(img, noises) 
    test_av_blur(img, noises)
    test_av_gauss(img, noises) 
    test_median(img, noises)    
    test_stacking(img, noises)  
  
    
if __name__ == '__main__':
    main(sys.argv)