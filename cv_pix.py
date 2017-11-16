# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:13:41 2017

@author: rd0348
"""

import cv2
import numpy as np

img_file = 'panda.jpg'
img = cv2.imread(img_file,cv2.IMREAD_COLOR)
alpha_img = cv2.imread(img_file,cv2.IMREAD_UNCHANGED)
gray_img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)

print(type(img))
print('RGB shape : ',img.shape)
print('ARGB shape : ',alpha_img.shape)
print('Gray shape : ',gray_img.shape)
print('img.dtype : ',img.dtype)
print('img.size : ',img.size)

