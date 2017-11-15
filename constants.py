# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:00:13 2017

@author: rd0348
"""

import tensorflow as tf
word=tf.constant('hello tensorflow!')
with tf.Session() as sess:
    print(sess.run(word))