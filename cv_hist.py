# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:58:11 2017

@author: rd0348
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('panda.jpg',-1)
cv2.imshow('GoldenGate',img)

color = ('r','g','b')
for channel,col in enumerate(color):
    histr=cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0.256])

plt.title('histogram for color scale picture')
plt.show()

