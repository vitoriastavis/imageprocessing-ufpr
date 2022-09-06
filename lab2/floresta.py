# Lab 02 - Segmentacao com cores
# Vitória Stavis de Araujo - GRR20200243

# importar modulos necessarios
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

# segmenta a imagem, mostra e salva o resultado
def segmentation(img, hsv_img, out_name, l_green, d_green):

    # quadrado das cores para plotar
    lg_square = np.full((10,10,3), l_green, dtype = np.uint8)/255.0
    dg_square = np.full((10,10,3), d_green, dtype = np.uint8)/255.0

    # plot das cores
    plt.subplot(1,2,1)
    plt.imshow(hsv_to_rgb(lg_square))
    plt.subplot(1,2,2)
    plt.imshow(hsv_to_rgb(dg_square))
    plt.show()

    # cria mascara para segmentar
    mask = cv.inRange(hsv_img, d_green, l_green)

    result = cv.bitwise_and(img, img, mask = mask)

    # plot da mascara e da imagem segmentada
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

    # salva imagem segmentada
    plt.imsave(out_name, result)

def color_gen(h, s, v):

    # limites inferiores e superios do hue, para manter em verde
    # divisao em dois para escala do opencv 0->180
    upper = 160/2
    lower = 82/2

    # encontra valores de h que se encaixam no verde
    h_green = h[h < upper]
    h_green = h_green[h_green > lower]
    h_green

    # valores medios
    h_mean = h_green.mean()
    s_mean = s.mean()
    v_mean = v.mean()

    # desvio padrao
    h_sd = np.std(h_green)
    s_sd = np.std(s)
    v_sd = np.std(v) 

    # agora vamos encontrar as duas cores
    # somando desvio padrao na media e normalizando

    # light green hue
    l_h = h_mean+h_sd

    # light green saturation
    l_s = s_mean+2*s_sd
    if l_s > 255:
        l_s = 255
    if l_s < 0:
        l_s = 5

    # light green value
    l_v = v_mean+3*v_sd
    if l_v > 255:
        l_v = 255
    if l_v < 0:
        l_v = 25

    # dark green hue
    d_h = h_mean-2*h_sd

    # dark green saturation
    d_s = s_mean-3*s_sd
    if d_s > 255:
        d_s = 255
    if d_s < 0:
        d_s = 50

    # dark green value
    d_v = v_mean-2*v_sd
    if d_v > 255:
        d_v = 255
    if d_v < 0:
        d_v = 25

    # salvando light e dark green
    l_green = (l_h, l_s, l_v)
    d_green= (d_h, d_s, d_v)

    return (l_green, d_green)    

# cria e mostra o grafico 3d do hue saturation e value da imagem
# retorna os arrays de h, s e v
def color_graph(img, hsv_img):

    # separar cores para fazer grafico
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin = -1., vmax = 1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # separar hue saturation e value
    h, s, v = cv.split(hsv_img)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection = '3d')

    # grafico das cores
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker = '.')
    axis.set_xlabel('hue')
    axis.set_ylabel('saturation')
    axis.set_zlabel('value')     
    plt.show()   

    return (h, s, v)


def main(argv):
    
    # ler a imagem
    img = cv.imread(argv[1])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
  
    # valores de hue, saturation e value
    h, s, v = color_graph(img, hsv_img)

    # cores do threshold
    l_green, d_green = color_gen(h, s, v)

    # segmentacao da imagem
    segmentation(img, hsv_img, argv[2], l_green, d_green)


if __name__ == '__main__':
    main(sys.argv)