""" Starter code for logistic regression model to solve OCR task 
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
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.000001
batch_size = 64
n_epochs = 25

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist

DATA_PATH = 'heart.csv'
BATCH_SIZE = 2
N_FEATURES = 9

filename_queue = tf.train.string_input_producer(["heart.csv"])
reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
_, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)
    # for this dataset, out of 9 feature columns, 
    # 8 of them are floats (some are integers, but to make our features homogenous, 
    # we consider them floats), and 1 is string (at position 5)
    # the last column corresponds to the lable is an integer

record_defaults = [[1.0] for _ in range(N_FEATURES)]
record_defaults[4] = ['']
record_defaults.append([1])

    # read in the 10 rows of data
content = tf.decode_csv(value, record_defaults=record_defaults) 

    # convert the 5th column (present/absent) to the binary value 0 and 1
content[4] = tf.cond(tf.equal(content[4], tf.constant('Present')), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    # pack all 9 features into a tensor
features = tf.stack(content[:N_FEATURES])

    # assign the last column to label
label = content[-1]

    # shuffle the data to generate BATCH_SIZE sample pairs
data_batch, label_batch = tf.train.batch([features, label], batch_size=300)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    features, labels = sess.run([data_batch, label_batch])
    print(labels)
    coord.request_stop()
    coord.join(threads)




# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# Features are of the type float, and labels are of the type int

X = tf.placeholder(tf.float32, [300,9], name='X')
Y = tf.placeholder(tf.float32, [300], name='Y')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y

w = tf.Variable(tf.zeros([9]), name='w')
b = tf.Variable(0.0, name='b')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE

logits = tf.tensordot(X,w,1) + b

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')

# Step 6: define training op
# using gradient descent to minimize loss

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	

	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(1):

			_, l=sess.run([opt,loss], feed_dict={X: features, Y:labels})
			
			total_loss += sum(l)
		print('loss epoch {0}: {1}'.format(i, total_loss))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') ; w,b=sess.run([w,b]) ; print('w:{0}, b:{1}'.format(w,b))

	# test the model
	preds = tf.nn.sigmoid(logits)
	correct_preds = tf.equal(tf.round(preds), Y)
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
	


	total_correct_preds = 0 ; temp=sess.run(tf.round(preds), feed_dict={X: features, Y:labels}); print(temp)
	
	for i in range(1):
		accuracy_batch = sess.run([accuracy], feed_dict={X: features, Y:labels});
		total_correct_preds += accuracy_batch[0]	
	
	print('Accuracy {0}'.format(total_correct_preds/300))
