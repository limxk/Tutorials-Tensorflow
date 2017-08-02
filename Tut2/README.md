Task :
- Implement Logistic Regression on MNIST dataset (03_logistic_regression_mnist_starter.py)
- Complete simple exercises on Tensorflow (e01.py), as a precursor to tutorial 3 (assignment 1 in the stanford module)

Key Findings :
- Read up on softmax vs sigmoid function : previously (from Andrew Ng's machine learning course) I learnt only the sigmoid function to apply to 2-class logistic regression, then extended it further to multi-class logsitic regression without the use of softmax function. Softmax allows us to do this in a single line.
- Read up on mini-batch gradient descent : in all previous exercises the dataset was generally small, hence there is no need to balance efficiency with accuracy, and standard gradient descent was always used. With the MNIST dataset, I implemented mini-batch gradient descent for the first time, and I played around with the batch size to investigate its effect on efficiency vs accuracy.
