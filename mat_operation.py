# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:09:00 2017

@author: rd0348
"""

import tensorflow as tf
diagonal = tf.constant([1,2,3,4])
with tf.Session() as sess:
    print(tf.diag(diagonal))

