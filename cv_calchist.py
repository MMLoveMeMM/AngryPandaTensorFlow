# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:47:40 2017

@author: rd0348
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_img = cv2.imread('panda.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('GoldenGate',gray_img)
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.hist(gray_img.ravel(),256,[0,256])
plt.title('Histogram for gray scale picture')
plt.show()
