# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:30:19 2017

@author: rd0348
"""

import tensorflow as tf

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v",[1])
    
print("sp:",sp.name)
print("var1:",var1.name)

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v",[1])
    
with tf.variable_scope(sp) as sp1:
    var3 = tf.get_variable("v3",[1])
    
print("sp1:",sp1.name)
print("var2:",var2.name)
print("var3:",var3.name)