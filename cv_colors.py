# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:59:14 2017

@author: rd0348
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('panda.jpg')
gray_img = cv2.cvtColor(bgr_img,cv2.COLOR_RGB2GRAY)
cv2.imwrite('san_francisco_grayscale.jpg',gray_img)

plt.imshow(gray_img,cmap = plt.get_cmap('gray'))
plt.xticks([]),plt.yticks([])
plt.show()

# ------------------------------------------------------
b,g,r = cv2.split(bgr_img)
rgb_img = cv2.merge([r,g,b])
plt.imshow(rgb_img)
plt.xticks([]),plt.yticks([])
plt.show()

# RGB inverted plot
