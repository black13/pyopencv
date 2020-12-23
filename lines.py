import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng

#https://stackoverflow.com/questions/48615935/merging-regions-in-mser-for-identifying-text-lines-in-ocr

image = cv2.imread('/home/jjosburn/temp/pytorch_models/images/outputname-0017.png',cv2.IMREAD_GRAYSCALE)
(thresh, img_bin) = cv2.threshold(image, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
img_bin = 255-img_bin 
kernel_length = np.array(image).shape[1]//80
 
# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))




# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

# now show large contour
drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
cnts = cv2.findContours(verticle_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv2.drawContours(img_bin, [c], -1, (0,0,0), 3)

boundRect = cv2.boundingRect(cv2.convexHull(cnts[0])) 
print(boundRect)
height, width = image.shape
width_cutoff = width // 2
s1 = img_bin[boundRect[1]:, :boundRect[0]]
s1 = 255-s1 
s2 = img_bin[boundRect[1]:, boundRect[0]:]
s2 = 255-s2 
s3 = img_bin[:boundRect[1],:]
s3 = 255-s3
for i in range(len(cnts)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    hull = cv2.convexHull(cnts[i]) 
    cv2.drawContours(drawing, cnts, i, color)
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, hull, i, color)
    boundRect = cv2.boundingRect(hull)
    
    print(hull)


cv2.imwrite("contours.png", drawing) 

height = image.shape[0]
width = image.shape[1]

# Cut the image in half

plt.subplot(131),plt.imshow(s1, cmap = 'gray')
plt.title('vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(s2, cmap = 'gray')
plt.title('horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(s3, cmap = 'gray')
plt.title('horizontal'), plt.xticks([]), plt.yticks([])

plt.show()


