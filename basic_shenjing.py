# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:19:48 2017

@author: rd0348
"""

import tensorflow as tf
import numpy as np

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # random_normal 函数说明 :正太分布随机数，均值mean,标准差stddev 
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # zeros 函数说明 : 产生一个以1填充的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 公式 : inputs * Weight + biases
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 1.训练的数据
# Make up some real data 
# linspace 函数说明 : 生成等差数列 
# 参数说明 : 第一个参数 : 数列第一个起始数;第二个参数 : 数列最后一个数;第三个参数 : 等差数个数,默认是50个
x_data = np.linspace(-1,1,300)[:, np.newaxis]
# random.normal 函数说明 : 高斯分布函数
# 参数说明 : 第一个参数 : u; 第二个参数 : sigama; 第三个参数 : 输出的shape，默认为None，只输出一个值
# http://blog.csdn.net/lanchunhui/article/details/50163669
noise = np.random.normal(0, 0.05, x_data.shape)
# square : 函数说明 : 平方
# pow : 乘方
# sqrt : 平方根
y_data = np.square(x_data) - 0.5 + noise

# 2.定义节点准备接收数据
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
# addd_layer : 第四个参数 : tf激活函数
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
# 公式添加在layer中
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data  
# 交叉熵评估代价
# http://blog.csdn.net/qq_16137569/article/details/72568793  
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小                   
# 这一行定义了用什么方式去减少 loss，学习率是 0.1 
# 优化器Optimizer , 下面是其子类
# 对于梯度的优化，也就是说，优化器最后其实就是各种对于梯度下降算法的优化
# http://blog.csdn.net/xierhacker/article/details/53174558      
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
