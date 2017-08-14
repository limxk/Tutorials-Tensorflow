Task :
- Implement word2vec. There wasn't much room for improvisation for this, and I largely followed the example given.

- Also explored an alternative (slightly different) implementation of word2vec, from the word2vec website you introduced ( http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/ ). Placed it under /alt_word2vec_udacity . The code is easier to follow, with a lot of checks in place while running. I'm more likely to use this code as a base if I were to develop a word2vec project further.


Main idea of word2vec (please correct me if im wrong anywhere here) :

- word2vec is based on a basic idea that two words are more similar / related, if they frequently appear within a short vicinity (of a fixed number of words) of each other.

- It can be thought of as the probability of each target word appearing within the short vicinity of a center word (skip-gram), or vice versa, the probability of a center word appearing within the short vicinity of a few target words (CBOW). Both ideas have slight variations in their algorithms and efficiency/accuracy, but are largely similar.

- The main idea of implementing word2vec is by representing each word as a vector of d dimensions (where d is set by the user). Known word pairs which appear within the vicinity of each other are then randomly chosen to form a training set of a certain batch size, and the training set is run through an optimizer to update the weights of each word-vector to maximise the similarity between word pairs. This training step is repeated a large number of times, like 10000 and above.

- Within the algorithm, the softmax function is used to estimate the probability, and is required when updating weights. The softmax function is highly inefficient in this case however, due to the large dimensions of the system (equivalent to the total number of known words). To combat this, we instead use either negative sampling or NCE while updating weights. (Softmax is still used to predict probabilities after training)

- Negative sampling involves the idea of NOT applying softmax to optimize the system, but instead just update the values of the most negative weights in each iteration. This is much faster, but does not guarantee convergence. NCE is a modified version of negative sampling, which is done in such a way that softmax is still not used (hence efficient), yet convergence is guaranteed.

Question :

- I understand that word2vec can be used to effectively find the similarity between words, but how is that applied elsewhere, in search for example? If "good" is most similar to "great" via word2vec, does that mean that when a query has the word "good", we also include the word "great" in the query? (Sounds too simplistic to me.) Or is there still no well-established optimal way to apply word2vec (to search) yet?
