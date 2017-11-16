# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:48:42 2017

@author: rd0348
"""

import tensorflow as tf
# placeholder 占位符,按照程序变量声明,但是没有付初始值更好理解
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict = {input1:[7.],input2: [2.]}))#输出的时候才赋值
# ----------------------------------------------------------------------------------
a = tf.constant(5,name="input_a")
b = tf.constant(3,name="input_b")
c = tf.multiply(a,b,name="mul_c")
d = tf.add(a,b,name="add_d")
e = tf.add(c,d,name="add_e")

sess = tf.Session()
print("result e = ",sess.run(e))
sess.close()
# ---------------------------------------------------------------------------------
a1 = tf.zeros([2,3],tf.int32)
b1 = tf.ones([2,3],tf.int32)

a2 = tf.random_normal([5,5],mean=0.0,stddev=1.0)
b2 = tf.random_uniform([5,5],minval=0,maxval=1)

# ------------------------------------------------------------------------------
a3 = tf.placeholder(tf.int32,[3.])
b3 = tf.constant([1,1,1])

c3 = a3+b3

with tf.Session() as sess:
    print(sess.run(c3,feed_dict={a3:[1,2,3]}))
    
# --------------------------------------------------------------------------------
a4 = tf.Variable([1,1])
b4 = tf.Variable([2,2])
assign_op = a4.assign(b4) # a4 = b4

init = tf.global_variables_initializer(); # 初始化有变量的算子
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    sess.run(assign_op)
    print(sess.run(a))

# -----------------------------------------------------------------------------
x = tf.constant([3.0,1.0])
y = tf.constant([1.0,2.0])
z = x*y+x*x
dx,dy = tf.gradients(z,[x,y])

with tf.Session() as sess:
    dx_v,dy_v=sess.run([dx,dy])
    print(dx_v)
    print(dy_v)



