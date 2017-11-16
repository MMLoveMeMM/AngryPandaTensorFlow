# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:54:13 2017

@author: rd0348
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('panda.jpg',0)
edges=cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('edge image'),plt.xticks([]),plt.yticks([])

plt.show()