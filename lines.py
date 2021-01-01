import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng
from skimage.morphology import skeletonize

def imageabovehorizontal(image):
    img_bin = 255-image 
    kernel_length = np.array(image).shape[1]//80
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    cnts = cv2.findContours(horizontal_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boundRect = cv2.boundingRect(cv2.convexHull(cnts[0])) 
    
    ret = img_bin[:boundRect[1],:]
    #ret = 255-ret
    return ret

def imageleftrightvertical(image):
 
    img_bin = 255-image 
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

image = cv2.imread('/Users/jjosburn/Documents/programming/dictionary_heath/bruel/file-0018.png',cv2.IMREAD_GRAYSCALE)
(thresh, img_bin) = cv2.threshold(image, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#ret = imageabovehorizontal(img_bin)

#ret = skeletonize(ret)

#plt.imshow(ret,cmap = 'gray')
#plt.show()


ret = imageleftrightvertical(img_bin)
plt.imshow(ret[0],cmap = 'gray')
plt.show()


horizontal = 255-np.copy(ret[0]) 
horizontal_size = 10
# Structure element for extracting horizontal lines (100 pixels wide 1 pixel tall)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.dilate(horizontal, horizontalStructure)
plt.imshow(horizontal, cmap="gray")
plt.show()