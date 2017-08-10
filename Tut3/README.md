Task :
- Assign1, Problem2, Task2 : Implement Logistic Regression on notMNIST dataset (notmnist.py). Nothing particularly interesting, most of the code is already given, so it's basically a copy and paste
- While tinkering around however, I uncoevered several points of confusion, not sure whether they are iumportant :

Question #1 : Vanilla Gradient Descent (VGD) vs Stochastic Gradient Descent (SGD)
While working on the problem, I noticed that the linear regression in Tut 1 had this code :
	
	for x, y in data:
		_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
  
- Why do we input the data into the optimizer one by one? Is this implementing SGD? Or am I mistaken?

I did some experimentation on running the data by feeding it in one by one (linear_regression_sgd.py), and putting the data in as an entire batch (linear_regression_vgd.py). The final values of w and b are similar, with a certain variance. However, other parameters changed quite a bit.

- Why does the optimal learning rate to ensure convergence vary greatly in each case? In particular, the converging learning rate for VGD is much smaller than that of SGD

- Why does changing the learning rate also change the total loss in each epoch greatly? 
(Eg : for SGD, learning rate = 0.001 gave a total loss of epoch 0 as 2069, while learnig rate = 0.0001 gave a total loss of epoch 0 as 845)


Question#2 : Batch Gradient Descent (BGD)
The logistic regression in Tut 2 implemented batch gradient descent.

- My understanding of BGD is that it sacrifices speed at the expense of greater noise in updates. With a smaller batch size, the optimizer updates the parameters over a smaller batch of data, but has to iterate over more batches to cover the entire training set, and hence the updates can be noisier. With a larger batch size, there will be less iterations, but the update will be done slower, since it has to perform calculations over a larger batch of data at once. Is my understanding correct?

- Is a batch size of 1 equivalent to SGD?

- I tried to play around the batch size in logistic_regression_bgd.py , and I noticed something weird. The updates per epoch is run the fastest for a moderate value of batch size (say 64 or 128), but when I set the batch size to a really small number (like 2), the epoch runs really slowly. This is counterintuitive to my understanding that a smaller batch size has greater speed. Am I mistaken anywhere?