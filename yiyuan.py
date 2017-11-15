# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:27:11 2017

@author: rd0348
"""

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
        Train_loss=tf.reduce_mean(tf.pow((Y_true-Y_pred),2))/2
    #梯度下降优化器
    with tf.name_scope('Train'):
        Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
        #训练节点
        TrainOp=Optimizer.minimize(Train_loss)
    with tf.name_scope('Eval'):
        #评估节点
        EvalLoss=tf.reduce_mean(tf.pow((Y_true-Y_pred),2))/2
    writer=tf.summary.FileWriter("logs/test_tensorboard",tf.get_default_graph())
    writer.close()