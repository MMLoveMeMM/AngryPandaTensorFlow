# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:14:23 2017

@author: rd0348
"""

import numpy as np
import tensorflow as tf
from sklearn import linear_model
from sklearn import preprocessing
x_data = np.loadtxt("ex3x.dat").astype(np.float32)
y_data = np.loadtxt("ex3y.dat").astype(np.float32)
reg = linear_model.LinearRegression()
reg.fit(x_data, y_data)
print("Coefficients of sklearn: K=%s, b=%f" % (reg.coef_, reg.intercept_))
scaler = preprocessing.StandardScaler().fit(x_data)
print(scaler.mean_, scaler.scale_)
x_data_standard = scaler.transform(x_data)
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))
y = tf.matmul(x_data_standard, W) + b
loss = tf.reduce_mean(tf.square(y - y_data.reshape(-1, 1)))/2
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W).flatten(), sess.run(b).flatten())
print("Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (sess.run(W).flatten(), sess.run(b).flatten()))
print("Coefficients of tensorflow (raw input): K=%s, b=%s" % (sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W))))