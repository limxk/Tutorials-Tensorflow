""" Simple logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import csv

# Define paramaters for the model
learning_rate = 0.00001
#batch_size = 128
train_size = 300
n_epochs = 50

# Step 1: Read in data

DATA_FILE = 'heart.csv'



Xtrainset=[]
Ytrainset=[]
testset=[]

with open(DATA_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        
        if row['famhist']=='Present':
            row['famhist']=100
        else:
            row['famhist']=0
        
        temparray=[row['sbp'],row['tobacco'],row['ldl'],row['adiposity'],row['famhist'],row['typea'],row['obesity'],row['alcohol'],row['age']]
        
        if len(Xtrainset) < train_size:
            #trainset.append([temparray, row['chd']])
            Ytrainset.append(row['chd'])
            Xtrainset.append(temparray)
        else:
            testset.append([temparray, row['chd']])




# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# each lable is one hot vector.
X = tf.placeholder(tf.float32, [train_size,9], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [train_size], name='Y_placeholder')
Xtest = tf.placeholder(tf.float32, [9], name='X_placeholder') 
Ytest = tf.placeholder(tf.float32, name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.zeros(shape=[9]), name='weights')
b = tf.Variable(0.0, name="bias")

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through sigmoid layer
logits = tf.tensordot(X, w, 1) + b


# Step 5: define loss function

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	# to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())	

    for i in range(n_epochs): # train the model n_epochs times
        total_loss = 0

        for _ in range(1): #for x,y in trainset: 
            _, l = sess.run([optimizer, loss], feed_dict={X: Xtrainset, Y:Ytrainset})
            total_loss += sum(l)
        print('Average loss epoch {0}: {1}'.format(i, total_loss))
	

    print('Optimization Finished!') ; print('Total time: {0} seconds'.format(time.time() - start_time))

    w,b=sess.run([w,b]) 
    print('w:{0}, b:{1}'.format(w,b))
    
    # test the model

    preds = tf.nn.sigmoid(logits)
    correct_preds = tf.equal(tf.round(preds), Y)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
	
    total_correct_preds = 0
    
    for _ in range(1):
        estimate = sess.run(preds, feed_dict={X: Xtrainset, Y:Ytrainset}) ; print(estimate)
        accuracy_temp = sess.run([accuracy], feed_dict={X: Xtrainset, Y:Ytrainset}) 
        total_correct_preds += accuracy_temp[0]

    
    print('Accuracy {0}'.format(total_correct_preds/len(Ytrainset)))

    writer.close()
