# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:03:08 2017

@author: rd0348
"""

import tensorflow as tf
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print ('a+b=',sess.run(add,feed_dict={a:2,b:3}))
    print ('a*b=',sess.run(mul,feed_dict={a:2,b:3}))