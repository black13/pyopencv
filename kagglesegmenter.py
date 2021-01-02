import cv2
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.signal import argrelmin
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import tensorflow as tf

#pip3 install opencv-python==4.1.2.30 seems to work best on ubuntu.

warnings.filterwarnings('ignore')

def showImg(img, cmap=None):
    plt.imshow(img, cmap=cmap, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def imageleftrightvertical(image):
    (thresh, img_bin) = cv2.threshold(image, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
    img_bin = 255-img_bin 
    kernel_length = np.array(image).shape[1]//80
    
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
     
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    cnts = cv2.findContours(verticle_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    boundRect = cv2.boundingRect(cv2.convexHull(cnts[0])) 
    print(boundRect)
    height, width = image.shape
    width_cutoff = width // 2
    left = img_bin[boundRect[1]:, :boundRect[0]]
    left = 255-left 
    right = img_bin[boundRect[1]:, boundRect[0]:]
    right = 255-right 
    return (left,right)

def applySummFunctin(img):
    res = np.sum(img, axis = 0)    #  summ elements in columns
    return res    

def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def smooth(x, window_len=11, window='hanning'):
    #     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.") 
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.") 
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") 
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if not orient in ['vertical', 'horizontal']:
        raise ValueError("Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'") 
    if orient == 'vertical': 
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    else:
            for i, l in enumerate(lines_arr):
                line = l
                plt.subplot(40, 1, i+1)  # A grid of 40 rows x 1 columns
                plt.axis('off')
                plt.title("Line #{0}".format(i))
                _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        print("x1=", x1, ", x2=", x2, ", Diff= ", x2-x1)
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines
       
def main():
    image = cv2.imread('/home/jjosburn/temp/pytorch_models/images/outputname-0018.png',cv2.IMREAD_GRAYSCALE)
    
    ret=imageleftrightvertical(image)
    
    img3 = np.transpose(ret[0])

    print(type(ret[0]))
    #showImg(img1, cmap='gray')
    #plt.imshow(img3,cmap = 'gray')
    #plt.show()

    
    kernelSize=9
    sigma=4
    theta=1.5
    
    
    imgFiltered1 = cv2.filter2D(img3, -1, createKernel(kernelSize, sigma, theta), borderType=cv2.BORDER_REPLICATE)
    img4 = normalize(imgFiltered1)
    
    #plt.imshow(img4,cmap = 'gray')
    #plt.show()
    
    
    (m, s) = cv2.meanStdDev(imgFiltered1)
    print(m[0][0])
    
    
    summ = applySummFunctin(img4)
    print(summ.ndim)
    print(summ.shape)
    
    # Find peaks(max).
    peak_indexes = argrelextrema(summ, np.greater)
    peak_indexes = peak_indexes[0]
    
    # Find valleys(min).
    valley_indexes = argrelextrema(summ, np.less)
    valley_indexes = valley_indexes[0]
    

    peak_x = peak_indexes
    peak_y = summ[peak_indexes]

    valley_x = valley_indexes
    valley_y = summ[valley_indexes]

    
    left_image = cv2.cvtColor(ret[0], cv2.COLOR_GRAY2BGR)
    H,W = left_image.shape[:2]

    for  y in peak_indexes:
        #print(y)
        cv2.line(left_image, (0,y), (W, y), (255,0,0), 1) 

    cv2.imwrite("result.png", left_image)
    '''
    # Plot main graph.
    (fig, ax) = plt.subplots()
    #ax.plot(data_x, data_y)
    
    # Plot peaks.
    peak_x = peak_indexes
    peak_y = summ[peak_indexes]

    valley_x = valley_indexes
    valley_y = summ[valley_indexes]

    ax.plot(summ)
    ax.plot(peak_x, peak_y, marker='o', linestyle='dashed', color='green', label="Peaks")
    
    # Plot valleys.

    ax.plot(valley_x, valley_y, marker='o', linestyle='dashed', color='red', label="Valleys")
    
    
    # Save graph to file.
    plt.title('Find peaks and valleys using argrelextrema()')
    plt.legend(loc='best')
    plt.show()
    #plt.savefig('argrelextrema.png')
'''
'''
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    smoothed = smooth(summ, 35)
    plt.plot(smoothed)
    plt.show()
    
    mins = argrelmin(smoothed, order=2)
    arr_mins = np.array(mins)
    plt.plot(smoothed)
    plt.plot(arr_mins, smoothed[arr_mins], "x")
    plt.show() 
    
    found_lines = crop_text_to_lines(img3, arr_mins[0])
     
    #sess = tf.Session()
    sess = tf.compat.v1.Session()
    found_lines_arr = []
    with sess.as_default():
        for i in range(len(found_lines)-1):
            found_lines_arr.append(tf.compat.v1.expand_dims(found_lines[i], -1).eval())
    
    display_lines(found_lines) 
    '''  
#25, 0.8, 3.5
if __name__ == "__main__":
    main()