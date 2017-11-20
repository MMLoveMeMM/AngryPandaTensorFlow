# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:32:20 2017

@author: rd0348
"""

import tensorflow as tf
import numpy as np

# 使用numpy 随机产生 100个数据
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100,0.200],x_data)+0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1])
# random_uniform : 函数说明 : 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
# 参数说明 : 第一个参数 : 采样下界,float,默认为0; 第二个参数 : 采样上界,默认值为0; 第三个参数 : 输出样本数目,int或者元组tuple
# embeddings = tf.Variable(
#        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# 构造一个vocabulary_size x embedding_size的矩阵，作为embeddings容器，
# 有vocabulary_size个容量为embedding_size的向量，每个向量代表一个vocabulary，
# 每个向量的中的分量的值都在-1到1之间随机分布

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 推到模型的公式
y = tf.matmul(W,x_data)+b

# 最小化方差
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

#启动默认图
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0,201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(W),sess.run(b))

