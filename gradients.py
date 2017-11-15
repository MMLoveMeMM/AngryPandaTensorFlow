# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:30:22 2017

@author: rd0348
"""

import tensorflow as tf
x=tf.constant([2.0,1.0])
y=tf.constant([1.0,2.0])
z=x*y+x*x

dx,dy=tf.gradients(z,[x,y])

with tf.Session() as sess:
    dx_v,dy_v=sess.run([dx,dy])
    print(dx_v)
    print(dy_v)