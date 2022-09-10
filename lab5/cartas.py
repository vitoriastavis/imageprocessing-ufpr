# Lab 05 - Identificação de linhas e palavras em imagens de cartas
# Vitória Stavis de Araujo - GRR20200243

# import required packages
import os
import sys
import cv2 as cv
from cv2 import BORDER_REPLICATE
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import find_peaks

# verifies arguments to be parsed to main program
def args_manager(args):
    
    num_args = len(args)
    
    if (num_args == 2):            
        if(args[1] == '-l'):
            return 0
        elif(args[1] == '-w'):
            return 1
        else:
            print("Please use -l to count lines and/or -w to count words")
            exit(1)
    elif (num_args == 3):
        if(args[1] == '-l' & args[1] == '-w'):
            return 2
        else:
            print("Please use -l to count lines and/or -w to count words")
            exit(1)
       

# function finds the corners given the top,
#  bottom, left, and right maximum pixels
def find_corners(bound):
    
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

# find the area of a contour
def find_area(c1):
    return abs(c1[0][0]-c1[1][0])*abs(c1[0][1]-c1[3][1])

# calculate and returns best angle score
def calc_angle_score(arr, angle):
    
    data = rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1, dtype=float)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return score

# calculate and returns rotated image with the best angle
def adjust_rotation(image, delta=3, limit=6):
    
    gray = image.copy()
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = calc_angle_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    return corrected
        
# morph open with kernel of dimensions (h x w) and iterations
def morph_open(img, h=10, w=10, iter=1):
    
    kernel = np.ones((h, w), np.uint8)
    img_open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iter)
     
    return img_open

# crops image based on height, width or both
# height and width can be set from 0 to 100
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

# returns how many peaks were found on the histogram
def get_num_peaks(hist):
        
    peaks, _ = find_peaks(hist)

    median = np.median(hist)
    pHeight = median

    peaks, _ = find_peaks(hist, height=pHeight, distance=83)

    return peaks.size

# returns normalized histogram
def get_hist(img):
    
    hist = np.sum(img, 1)
    hist_norm = hist / np.linalg.norm(hist)
    
    return hist_norm

# returns number of lines
def line_counter(name):
      
    img = crop_image(name, height=97, width=65) 
    img_morph = morph_open(img)
    
    thresh = cv.threshold(img_morph, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    
    hist = get_hist(thresh)
    
    n_lines = get_num_peaks(hist)

    return n_lines
    
# return number of words from the image of a text
def word_counter(img):
    
    # performing OTSU threshold
    _, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        
    # specify structure shape and kernel size 
    rectangle = cv.getStructuringElement(cv.MORPH_RECT, (17, 17))

    # applying dilation on the threshold image
    dilation = cv.dilate(thresh, rectangle, iterations = 1)

    # finding contours
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL,
                                                    cv.CHAIN_APPROX_NONE)

    #print(len(contours),'\n')
    
    # we've found the countours, now let's remove outliers

    #holds bounding box of each countour
    bndingBx = []
    corners = []
        
    err = 2 #error value for minor/major axis ratio
        
    #list will hold the areas of each bounding boxes
    areas = []

    #find the rectangle around each contour
    for num in range(0,len(contours)):
            
        #make sure contour is for letter and not cavity
        if(hierarchy[0][num][3] == -1):
                
            left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
            right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
            top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
            bndingBx.append([top, right, bottom, left])
            
    #find the edges of each bounding box
    for bx in bndingBx:
        
        corners.append(find_corners(bx))
            
    #go through each corner and append its area to the list
    for corner in corners:
        
        area = find_area(corner)
        areas.append(area)

    areas = np.asarray(areas) #organize list into array format
    avgarea = np.mean(areas) #find average area
    stdareas = np.std(areas) #find standard deviation of area
    outlier = (areas < (avgarea - (0.7*stdareas))) #find the outliers, these are probably the dots
    
    # image that will have the outliers removed
    clean = np.zeros((len(img), len(img[0])), np.uint8)

    for num in range(0, len(outlier)): #go through each outlier    
                            
            if(outlier[num]):
                
                black = np.zeros((len(img),len(img[0])),np.uint8)
                #add white pixels in the region that contains the outlier
                cv.rectangle(black,(corners[num][0][0],corners[num][0][1]),(corners[num][2][0],corners[num][2][1]),(255,255),-1)
                
                #perform bitwise operation on original image to isolate outlier                    
                test = cv.bitwise_and(thresh,black)    
                
                clean = clean + test
                
    clean = cv.bitwise_xor(thresh, clean)             
                               
    # applying dilation on the threshold image updated without outliers
    dilation2 = cv.dilate(clean, rectangle, iterations = 1)

    # finding contours
    contours2, _ = cv.findContours(dilation2, cv.RETR_EXTERNAL,
                                                    cv.CHAIN_APPROX_NONE)
     
    # creating a copy of image
    img2 = img.copy()   
      
    for cnt in contours2:
        
        x, y, w, h = cv.boundingRect(cnt)      
                 
        # drawing a rectangle on copied image        
        cv.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    #for cnt in contours:
            
    #    x, y, w, h = cv.boundingRect(cnt)      
                 
    #    # drawing a rectangle on copied image        
    #    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    #plt.subplot(1,2,1)        
    #plt.imshow(img, 'gray')
    #plt.subplot(1,2,2)
    plt.imshow(img2, 'gray')
    plt.show()
    plt.clf() 
    
    return len(contours2)  
    
def main(argv):
    
    n_correct = 0
    n_files = 0
    
    status = args_manager(argv)
                
    # current directory
    directory = os.getcwd()  

    # loop through files of the directory
    for filename in os.listdir(directory):
               
        _, ext = os.path.splitext(filename) 
                     
        if(ext == '.jpeg' or ext == '.jpg' or ext == '.jpeg'): 

            n_lines = 0
            n_words = 0
            
            n_files += 1
            
            # get number of lines of the current image 
            if(ext == '.jpeg'):
                lines = filename[-7:-5]
            else:
                lines = filename[-6:-4]
                
            # read image
            img = cv.imread(directory+'/'+filename)    
            
            # convert the image to gray scale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # adjust image rotation
            img_corrected = adjust_rotation(gray)

            if (status == 1 or status == 2):
                # find the number of words
                n_words = word_counter(img_corrected)     
                print(filename[:-5],':\n', n_words, 'words', '\n')
                
            elif (status == 0 or status == 2):
                # find the number of lines
                n_lines = line_counter(img_corrected)     
                
                if (int(lines) == int(n_lines)):                    
                        n_correct += 1
                    
                print(filename[:-5],':\n', n_lines, 'lines out of', lines, '\n')
                
    # return number of correct identified lines                          
    if(status == 0 or status == 2):
        print('correct line amount:', n_correct , '/' , n_files)
                    
if __name__ == '__main__':
    main(sys.argv)
    
    
