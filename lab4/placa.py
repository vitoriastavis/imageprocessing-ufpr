# Lab 04 - Melhora da qualidade da imagem
# Vitória Stavis de Araujo - GRR20200243

# importar modulos necessarios
import sys
from cv2 import equalizeHist
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib
from math import exp, pow, log
from scipy import stats

RADIUS = 30

## MEDIAN FILTER
def median(img, size):
    
    res = cv.medianBlur(img, size)
    return res
 
def get_mask(img, fshift):
    
    rows, cols = img.shape
     
    crow, ccol = rows/2 , cols/2

    n = len(fshift)
    m = 752  
    
    y, x = np.ogrid[-crow:n-crow, -ccol:m-ccol]
    
    mask = x * x + y * y <= RADIUS*RADIUS
            
    return mask 
 
def inverse_transform(masked_img):
        
    f_ishift = np.fft.ifftshift(masked_img)
    filtered_img = np.fft.ifft2(f_ishift)
    filtered_img = np.abs(filtered_img)
        
    return filtered_img 
 
def notch_filter(img):
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) 
    
    mask = get_mask(img, fshift)       
    
    masked = fshift * mask
    
    filtered = inverse_transform(masked)
    
    magnitude_spectrum = 100*np.log(1 + np.abs(masked))
    
    #plt.subplot(121)
    #plt.imshow(img, cmap = 'gray')
    #plt.subplot(122)   
    #plt.imshow(filtered, cmap = 'gray')
    #plt.show()
    
    return filtered
    
    
    
def main(argv):
    
    # ler a imagem
    img = cv.imread(argv[1], 0)     
       
    # calcular histograma
    hist = cv.calcHist([img], [0], None, [256], [0,256])   
    
    #notched = notch_filter(img)
    #print('imagem original', img.shape, type(img))
    #print('depois do processo', notched.shape, type(notched))    
               
    # método CLAHE
    clahe = cv.createCLAHE(clipLimit = 10.0, tileGridSize = (80, 80))
    # aplicar método na imagem que passou pelo filtro da mediana
    med_cl =  clahe.apply(median(img, 3))    
          
    # testando se vai precisar equalizar       
    skew = int(stats.describe(hist)[4])    
    if skew < 1:
        equ = equalizeHist(med_cl)      
        res = equ
    else:
        res = med_cl
    
    cv.imwrite(argv[2],res)    
       
    # salva a imagem resultante
    #plt.imsave(argv[2], res)
    
if __name__ == '__main__':
    main(sys.argv)
    
# https://stackoverflow.com/questions/37396419/find-proper-notch-filter-to-remove-noise-from-image
# https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python
# https://stackoverflow.com/questions/59921561/gaussian-notch-filter-in-python
# https://www.delftstack.com/howto/numpy/size-and-shape-of-array-in-python/


