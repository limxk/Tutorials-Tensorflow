Task :
- Implement a super simple Linear Regression using TensorFlow (TF), based on the notes and slides.
- Solutions were provided, but I tried my best to code the entire thing without looking at it.

Key Findings :
- Understood the basic workings of TF : first initialise all variables/functions, then run a session to conduct optimisation. I see how this is a preferred library --- the code is very clean, and it is very simple to conduct comparisons across different optimization variations.
- TF's gradient optimization does not require a gradient function input, but instead uses Automatic Differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation). It is a balance between Symbolic Differentiation and Numerical Differentiation, and essentially relies heavily on Chain Rule and the sequences of operations within the loss function to compute the gradient.
- I did some tinkering with the tutorial code itself, and tried to implement a simple investigation between square loss function and absolute loss function (as mentioned during our meeting). Both are comparable when used on dummy data with no outliers, but absolute loss function outperforms once huge outliers are present. However, absolute loss function has a disadvantage where multiple optimal solutions exist, as noted in http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/ 
- A Huber Loss function (https://en.wikipedia.org/wiki/Huber_loss) is already available to  combat this problem of outliers, but requires an input of a delta threshold. A possible improvement would be to consider the square root loss function instead, to be investigated.
