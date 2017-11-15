# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:35:53 2017

@author: rd0348
"""

import os
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with tf.Graph().as_default():
    #输入节点
    with tf.name_scope('Input'):
        X=tf.placeholder(tf.float32,shape=[None,784],name='X')
        Y_true=tf.placeholder(tf.float32,shape=[None,10],name='Y_true')
    #前向预测
    with tf.name_scope('Inference'):
        W=tf.Variable(tf.zeros([784,10]),name="Weight")
        b=tf.Variable(tf.zeros([10]),name="Bias")
        logits=tf.add(tf.matmul(X,W),b)
    #softmax把logistics变成概率分布
    with tf.name_scope('Softmax'):
        Y_pred=tf.nn.softmax(logits=logits)
    #损失节点
    with tf.name_scope('Loss'):
        TrainLoss=tf.reduce_mean(-tf.reduce_sum(Y_true*tf.log(Y_pred),axis=1))
    #训练节点
    with tf.name_scope('Train'):
        TrainStep=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(TrainLoss)
    #评估节点
    with tf.name_scope('Eval'):
        correct_prediction=tf.equal(tf.argmax(Y_pred,1),tf.argmax(Y_true,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    InitOp=tf.global_variables_initializer()
    writer=tf.summary.FileWriter(logdir='logs/mnist_softmax',graph=tf.get_default_graph())
    writer.close()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession()
    sess.run(InitOp)
    for step in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        _,train_loss=sess.run([TrainStep,TrainLoss],feed_dict={X: batch_xs, Y_true: batch_ys})
        if step%100==0:
            print("step",step,"loss",train_loss)
    #输出结果
    print("softmax_result:")
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_true: mnist.test.labels}))