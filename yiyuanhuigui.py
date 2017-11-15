# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:52:15 2017

@author: rd0348
"""

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from matplotlib import pylab as plt
#指定默认字体  
plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['font.family']='sans-serif'  
#解决负号'-'显示为方块的问题  
plt.rcParams['axes.unicode_minus'] = False

#产生训练数据集
train_X=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_train_samples=train_X.shape[0]
print('训练样本数量：',n_train_samples)
#产生测试样本
test_X=np.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
test_Y=np.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
n_test_samples=test_X.shape[0]
print('测试样本数量：',n_test_samples)
#计算图
with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X=tf.placeholder(tf.float32,name='X')
        Y_true=tf.placeholder(tf.float32,name='Y_true')
    with tf.name_scope('Inference'):
        W=tf.Variable(tf.zeros([1]),name='Weight')
        b=tf.Variable(tf.zeros([1]),name='Bias')
        Y_pred=tf.add(tf.multiply(X,W),b)
    with tf.name_scope('Loss'):
        #添加损失
        TrainLoss=tf.reduce_mean(tf.pow((Y_pred-Y_true),2))/2
    #梯度下降优化器
    with tf.name_scope('Train'):
        Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
        #训练节点
        TrainOp=Optimizer.minimize(TrainLoss)
    with tf.name_scope('Eval'):
        #评估节点
        EvalLoss=tf.reduce_mean(tf.pow((Y_pred-Y_true),2))/2
    #初始化节点
    InitOp=tf.global_variables_initializer()
    #存计算图
    writer=tf.summary.FileWriter(logdir="logs/test_tensorboard",graph=tf.get_default_graph())
    writer.close()
    #启动图
    sess=tf.Session()
    sess.run(InitOp)
    for step in range(1000):
        for tx,ty in zip(train_X,train_Y):
            _,train_loss,train_w,train_b=sess.run([TrainOp,TrainLoss,W,b],feed_dict={X:tx,Y_true:ty})
        if(step+1)%20==0:
            print("Train",'%04d' % (step+1),"loss","{:.5f}".format(train_loss))
        if(step+1)%100==0:
            test_loss=sess.run(EvalLoss,feed_dict={X:test_X,Y_true:test_Y})
            print("-Test","loss=","{:.9f}".format(train_loss),"W=",train_w,"b=",train_b)
    print("---End---")
    #绘图
    W,b=sess.run([W,b])
    train_loss=sess.run(TrainLoss,feed_dict={X:train_X,Y_true:train_Y})
    test_loss=sess.run(EvalLoss,feed_dict={X:test_X,Y_true:test_Y})
    print("Final W=",W," ,b=",b," ,trainloss=",train_loss," ,testloss=",test_loss)    
    
    plt.plot(train_X,train_Y,'ro',label='训练样本')
    plt.plot(test_X,test_Y,'b*',label='测试样本')
    plt.plot(train_X,W*train_X+b,label='拟合线')
    plt.legend()
    plt.show()