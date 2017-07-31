""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file

'''
# Original fire theft data from tutorial
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
'''

# Dummy data with Y ~ 3X and artificial outliers, to investigate square loss vs abs loss functions
X_input = np.linspace(-1, 1, 100)
Y_input = X_input * 3 + np.random.randn(X_input.shape[0]) * 0.5
Y_input[99]+=20
data = np.column_stack([X_input,Y_input])

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Both have the type float32

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


# Step 3: create weight and bias, initialized to 0
# name your variables w and b

v = tf.Variable(0.0, name='v')
w = tf.Variable(0.0, name='w')
b = tf.Variable(0.0, name='b')

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted

Y_predicted = w*X + b  #Linear Regression
#Y_predicted = v*X*X + w*X + b  #Quadratic Regression

# Step 5: use the square error as the loss function
# name your variable loss

losssq = tf.pow(Y - Y_predicted,2, name='loss')  
lossabs = tf.abs(Y - Y_predicted, name='loss')  

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss

optsq = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(losssq)
optabs = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(lossabs)

# Phase 2: Train our model
with tf.Session() as sess1:  #Train with square loss function
	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO	
    sess1.run(tf.global_variables_initializer())
	# Step 8: train the model
    for i in range(300): # run 300 epochs
        total_loss = 0
        for x, y in data:
			# Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
			# TO DO: write sess.run()
            _, l=sess1.run([optsq,losssq], feed_dict={X: x, Y:y})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss/n_samples))
	
    vsq,wsq,bsq=sess1.run([v,w,b])
 
    
with tf.Session() as sess2:  #Train with abs loss function
	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO	
    sess2.run(tf.global_variables_initializer())
	# Step 8: train the model
    for i in range(300): # run 300 epochs
        total_loss = 0
        for x, y in data:
			# Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
			# TO DO: write sess.run()
            _, l=sess2.run([optabs,lossabs], feed_dict={X: x, Y:y})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss/n_samples))
	
    vabs,wabs,babs=sess2.run([v,w,b])

    
# plot the results
print("vsq:{0} wsq:{1} bsq:{2}".format(vsq,wsq,bsq))
print("vabs:{0} wabs:{1} babs:{2}".format(vabs,wabs,babs))
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
#plt.plot(X, X * wsq + bsq, 'r', label='Predicted data')  #Linear regression plot
#plt.plot(X, vsq*X*X + X * wsq + bsq, 'ro', label='Predicted data')  #Quadratic regression plot
plt.legend()
plt.show()