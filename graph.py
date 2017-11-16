# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:37:31 2017

@author: rd0348
"""

import numpy as np
import tensorflow as tf

c = tf.constant(0.0)

g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)
    
g2 = tf.get_default_graph()
print(g2)
tf.reset_default_graph()
g3 = tf.get_default_graph()
print(g3)

print(c1.name)
t = g.get_tensor_by_name(name = "Const:0")
print(t)