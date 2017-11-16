# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:03:49 2017

@author: rd0348
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

x = np.random.randint(25,100,25)   # 25 randoms in (25,100)
y = np.random.randint(175,250,25)  # 25 randoms in (175,250)
z = np.hstack((x,y))               # z.shape : (50,)
z = z.reshape((50,1))              # reshape to a column vector
z = np.float32(z)


# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)

print('centers: %s' %centers)

A = z[labels==0]
B = z[labels==1]

# initial plot
plt.subplot(211)
plt.hist(z,256,[0,256])

# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.subplot(212)
plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')

plt.show()