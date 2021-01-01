

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

# read the input image 
image = cv2.imread('/Users/jjosburn/Documents/programming/dictionary_heath/bruel/file-0018.png',cv2.IMREAD_GRAYSCALE)
(thresh, img_bin) = cv2.threshold(image, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

# Display original image
plt.imshow(img_bin, cmap="gray")
plt.show()

# transform the image to binary with 0s and 1s
img = img_bin // 255

img = cv2.bitwise_not(img)


# transform the image to binary with 0s and 1s
img = img // 255

# invert the pixel values
# 0: background
# 1: foreground
img = cv2.bitwise_not(img)-254

n_pixels = img.shape[0] * img.shape[1]
print("Number of pixels:", n_pixels)
print("Image shape:",img.shape)
print("Unique values:",np.unique(img))

n_white_pixels = np.count_nonzero(img)
n_black_pixels = n_pixels - n_white_pixels
print("# foreground pixels:", n_white_pixels, "# background pixels:", n_black_pixels, "Image shape:",img.shape)

# Show binary image
plt.imshow(img, cmap="gray")
plt.show()

# create separate copies of the images
horizontal = np.copy(img)
vertical = np.copy(img)
print(np.unique(vertical))

# create horizontal structuring element
horizontal_size = 100
# Structure element for extracting horizontal lines (100 pixels wide 1 pixel tall)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))


# Apply morphology operations - Dilate
horizontal = cv2.dilate(horizontal, horizontalStructure)
plt.imshow(horizontal, cmap="gray")
plt.show()

# Erode the image
horizontal = cv2.erode(horizontal, horizontalStructure)

plt.imshow(horizontal, cmap="gray")
plt.show()

verticalsize = 200
# Structure element for extracting vertical lines (200 pixels tall and 1 pixel wide)
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

# Vertical dilation
vertical = cv2.dilate(vertical, verticalStructure)

# Vertical Erosion
vertical = cv2.erode(vertical, verticalStructure)

# Inverse vertical image
# vertical = cv2.bitwise_not(vertical)-254
plt.imshow(vertical, cmap="gray")
print(np.unique(vertical))

# Combine vertial and horizontal representations using an AND operation
merge = vertical * horizontal

# Close possible small holes using
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
closing = cv2.morphologyEx(merge, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap="gray")
plt.imsave("closing.png", closing, cmap="gray")

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)
print("# of boxes:",nlabels)

# Count pixel transitions
def count_transitions(img):
  assert len(np.unique(img)) <= 2, "Image is not binary"
  black_pixel = 0

  # template: white --> black
  template = [1,0]
  
  row_length = img.shape[1]
  if row_length % 2 != 0:
    row_length = row_length-1  
   
  transitions = 0
  for row in img:
    for i in range(row_length):
      if np.all(template == row[i:i+2]):
        transitions+=1
  return transitions

img = cv2.bitwise_not(img)-254
plt.imshow(img, cmap="gray")
plt.show()

# transform image to rgb to draw bboxes
rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[:,:,0] = img
rgb[:,:,1] = img
rgb[:,:,2] = img

valid_transition_ratio = []
valid_pixel_ratio = []
#  para cada retângulo envolvendo um objeto, calcule:
for count, box in enumerate(stats):
    
    l = box[0]
    t = box[1]
    r = box[0] + box[2]
    b = box[1] + box[3]
    
    # get the bbox
    img_patch = closing[t:b,l:r]
    real_patch = img[t:b,l:r]

    n_pixels_box = img_patch.shape[0] * img_patch.shape[1]
    if (n_pixels_box / n_pixels) > 0.9:
      continue
    
    # razao entre o numero de pixels pretos e o numero total de pixels (altura × largura)
    # assumindo numero total de pixels do patch (box)
    # 0 --> black pixel
    # 1 --> white pixel
    
    n_white_pixels = np.count_nonzero(real_patch)
    n_black_pixels = n_pixels_box - n_white_pixels
    print("# white pixels:", n_white_pixels, "# black pixels:", n_black_pixels, "Image shape:",img_patch.shape)
    pixel_ratio = n_black_pixels / n_pixels_box
    
    ht = count_transitions(real_patch)
    vt = count_transitions(real_patch.T)
                    
    transition_ratio = (ht+vt)/n_pixels_box     

    if (pixel_ratio >= 0.19 and pixel_ratio <= 0.38) and (transition_ratio >= 0.063 and transition_ratio < 0.13):
      plt.imsave("patch_" + str(count) + ".png", real_patch, cmap="gray")
      cv2.rectangle(rgb,(l,t),(r,b),(0,255,0),2)

      valid_pixel_ratio.append(pixel_ratio)
      valid_transition_ratio.append(transition_ratio)
   

# This image has to be rbg to print colored boxes
plt.figure(figsize=(17,17))
plt.imshow(rgb)
plt.imsave("lines_final.png", rgb)

print("Total of lines:", len(valid_pixel_ratio))