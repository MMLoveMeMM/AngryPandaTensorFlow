# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:21:52 2017

@author: rd0348
"""
#$ echo -e "Alpha1,A1\nAlpha2,A2\nAlpha3,A3" > A.csv  
#$ echo -e "Bee1,B1\nBee2,B2\nBee3,B3" > B.csv  
#$ echo -e "Sea1,C1\nSea2,C2\nSea3,C3" > C.csv  

import tensorflow as tf  
# 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
filenames = ['A.csv']  
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  
# 定义Reader  
reader = tf.TextLineReader()  
key, value = reader.read(filename_queue)  
# 定义Decoder  
record_defaults = [[1], [1], [1], [1], [1]]  
col1, col2, col3, col4, col5 = tf.decode_csv(value,record_defaults=record_defaults)  
features = tf.stack([col1, col2, col3])  
label = tf.stack([col4,col5])  
example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=2, capacity=200, min_after_dequeue=100, num_threads=2)  
# 运行Graph  
with tf.Session() as sess:  
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
    for i in range(1):  
        #e_val,l_val = sess.run([example_batch, label_batch])  
        #print(e_val,l_val)
        print(example_batch.eval(), label_batch.eval())
    coord.request_stop()  
    coord.join(threads) 