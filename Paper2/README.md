Summary :

This paper discusses two different common document retrieval models. The first model, the local model, is the traditional method of relying mainly on exact matching of query terms. This model can be improved to a certain extent by relying on ranking strategies (BM25, tf-idf, pagerank, etc), match positions (proximity and location of term matches in document), as well as rudimentary forms of inexact term matches (matching synonyms of query terms). 

The second model is the distributed model, whose main difference is in considering not only the query terms itself, but also the terms within its neighbourhood of the training set (skip-grams). Additionally, we can also tokenise each term (n-grams) to lower the dimensionality of the vector space of the training set. Through deep-learning, we can estimate the probability/frequency of a token appearing within the neighbourhood of a query term, and take that to be its similarity score with the query term. This allows us to perform document retrieval through inexact term matching effectively, matching not only synonyms, but also terms that frequently come together.

The paper combines both models by applying both methods on the same query, and adding both scores to get an overall score. The new method boasts of an improved performance over previous methods that is statistically significant via paired t-test (p<0.05).

---

Questions :

Why did the paper get accepted?
The paper combined several well-known techniques (tokenization, word embedding + similarity score, local model + distributed model) to come up with a new document retrieving technique that supposedly combines the advantages of each technique.
It also includes quite a number of tables, numbers and graphs to explain and verify its findings (is this important?)

How can I improve this paper?
The paper combines several common techniques, but simply adds/concatenates them together. If the techniques can be integrated together more closely, greater improvements to accuracy might be made. A simple example would be to get a weighted sum of both model scores, instead of just a direct sum. A difficult example would be to combine both models into a single algorithm, instead of just running them in tandem.

Can the techniques used in this paper be applied somewhere else?
In the Math Search technique that Prof Hui passed to me, a similar tokenization algorithm has already been applied to math formulae (as opposed to words). It might be possible to directly apply the techniques in this paper on math formulae (find a similarity score for each formula term, then use it to retrieve documents based on math formula queries).