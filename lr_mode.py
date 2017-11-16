# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:39:18 2017

@author: rd0348
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义输入与输出
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义模型参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10,]))

# 定义模型
output = tf.nn.xw_plus_b(x, W, b)
prob = tf.nn.softmax(output)

# 定义loss，采用交叉熵
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(prob), axis=1))

# 定义优化器
learning_rate = 1e-04
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 定义精确度
correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练参数
steps = 10000
batch_size = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化参数
    # 开始训练
    for i in range(steps):
        xs, ys = mnist.train.next_batch(batch_size)
        # 执行训练
        l = sess.run([train_op, cross_entropy], feed_dict={x:xs, y:ys})
        if i % 1000 == 0:
            print("steps %d",i)
            #print("Steps %d , loss: %f" % (i, l))
    # 测试集上测试
    print('the final result : ',sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print('Done')