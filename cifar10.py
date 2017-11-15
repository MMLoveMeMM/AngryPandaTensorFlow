# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:14:37 2017

@author: rd0348
"""

import tensorflow.models.image.cifar10.cifar10 as cifar10
cifar10.maybe_download_and_extract()
images, labels = cifar10.distorted_inputs()
print (images)
print (labels)