# importar modulos necessarios
import cv2 as cv
import os

# diretorio
directory = os.getcwd()

# vetor com quantidade acertada em cada método
# posicoes: correlation, chi-square, intersection e bhattacharyya distance
qty = [0,0,0,0]

# iterando pelos 4 metodos
for method in range(4): 

  value_f = 'file'     

  if method == 0:
    print("Método: Correlação")
    value = -1
  elif method == 1:
    print("Método: Chi-Square")
    value = 1000
  elif method == 2:   
    print("Método: Intersecção")
    value = -1
  else:
    print("Método: Distância Bhattacharyya")
    value = 1000

  # iterando pelos arquivos no diretorio
  for filename in os.listdir(directory):

    # ler a imagem modelo
    model = cv.imread(directory+'/'+filename, 0)  

    # gerar e normalizar histograma
    h_model = cv.calcHist([model], [0], None, [256],
                              [0,256])      
    
    cv.normalize(h_model, h_model, 0, 255, cv.NORM_MINMAX)     

    # iterando pelos arquivos que serao comparados
    for filename2 in os.listdir(directory):
      
      if filename != filename2:

        # ler imagem para ser testada
        test = cv.imread(directory+'/'+filename2, 0) 

        # gerar e normalizar histograma
        h_test = cv.calcHist([test], [0], None, [256],
                                    [0, 256])
        cv.normalize(h_test, h_test, 0, 255, cv.NORM_MINMAX)
                
        # comparacao dos dois histogramas
        score = cv.compareHist(h_model, h_test, method)  

        # print(filename2, ' ', score)

        # escolher maior score para correlation e intersection 
        # e score menor para chi-square e bhattachayya 

        # chi-square e bhattachayya 
        #print("metodo = ", method)
        if method == 1 or method == 3:
          if score < value:
            value = score
            value_f = filename2           
        else:
        # correlation e intersection
          if score > value:
            value = score
            value_f = filename2

    print('valuef = ',value_f)

    # todas as comparacoes de histogramas finalizadas
    # comparando o arquivo escolhido com o arquivo modelo
    if value_f[0] == filename[0] or value_f[0:2] == filename[0:2]:
      print(filename,': Acertou')
      qty[method] = qty[method] + 1     # atualizando vetor de qtde de acertos por metodo 
    else:
      print(filename,': Errou')

  print()

# printando porcentagem de acerto de cada método
print('-----------------------------------------')
print('Ácurária total Correlation: ', round((qty[0]/25*100),3),'%')
print('Ácurária total Chi-Square: ', round((qty[1]/25*100),3),'%')
print('Ácurária total Intersection: ', round((qty[2]/25*100),3),'%')
print('Ácurária total Bhattacharyya distance: ', round((qty[3]/25*100),3),'%')
