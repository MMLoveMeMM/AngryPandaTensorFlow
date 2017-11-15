# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:37:55 2017

@author: rd0348
"""

"""
计算步骤
1.算距离：给定测试样本特征向量，计算它与训练集中每个样本特征向量的距离
2.找近邻：圈定距离最近的k个训练样本，作为测试样本的近邻
3.做分类：根据这k个近邻归属的主要类别，来对测试样本分类
优化模型
没有显式的模型参数优化过程
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#导入MNIST数据集
mnist=input_data.read_data_sets("mnist-data/",one_hot=True)
#5000条数据用于训练，200用于测试
Xtrain,Ytrain=mnist.train.next_batch(5000)
Xtest,Ytest=mnist.test.next_batch(200)
print('Xtrain.shape:',Xtrain.shape,'Xtest,shape',Xtest.shape)
print('Ytrain.shape:',Ytrain.shape,'Ytest,shape',Ytest.shape)
#计算图占位符
xtrain=tf.placeholder("float",[None,784])
xtest=tf.placeholder("float",[784])
#计算距离
distance=tf.reduce_sum(tf.abs(tf.add(xtrain,tf.negative(xtest))),axis=1)
#预测：获得最小距离的索引 （根据最邻近的类标签进行判断）
pred=tf.arg_min(distance,0)
#评估：判断给定一条测试样本是否预测正确
accuracy=0
init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    Ntest=len(Xtest)
    for i in range(Ntest):
        #获取当前测试样本的最近邻
        nn_index=sess.run(pred,feed_dict={xtrain:Xtrain,xtest:Xtest[i,:]})
        #获得最近邻预测标签，与真实标签比较
        pred_class_label=np.argmax(Ytrain[nn_index])
        true_class_label=np.argmax(Ytest[i])
        print("Test",i,"Predicted Class Label:",pred_class_label,"True Class Label:",true_class_label)
        #计算准确率
        if pred_class_label==true_class_label:
            accuracy+=1
    print("Done")
    accuracy /=Ntest
    print("Accuracy:",accuracy)