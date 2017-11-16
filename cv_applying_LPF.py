# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:43:11 2017

@author: rd0348
"""

"""
In this section, we'll create a LPF to remove high frequency contents in the image. 
In other words, we're going to apply LPF to the image which has blurring effect.
从空域转换到频域,去掉高频部分,然后再反FFT,得到图片[也可以说压缩了图片]
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('panda.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows/2 , cols/2     # center

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()             
