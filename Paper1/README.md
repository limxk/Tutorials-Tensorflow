This paper investigates the identification of individuals engaged in self-harm through their flickr posts, and can be easily extended to other social media sources, such as facebook or instagram. The main technique involves scouring flickr posts and mining data from 4 categories : Textual (tags/words used bearing high similarity to self-harm content), Owner (characteristic of poster from his profile), Temporal (timing, frequency and regularity of posts), and Visual (aesthetic aspects of the post itself, such as its hue and saturation). Most of the conclusions derived from the analysis of each category are hardly surprising, with the only mildly interesting one being the discovery that the frequency of normal posts tend to be unimodal and peak in the afternoon (3 pm), while posts indicating self-harm are bimodal, with a similar peak at 3pm, and a smaller peak at 12mn.

The paper suggests two methods for predicting self-harm content in posts, one by supervised learning, which is possible if prior labelled data is available, and unsupervised learning otherwise. Regardless, both methods start off by vectorizing the data obtained from each of the 4 categories, and concatenating them to form a vector space. After that, supervised learning is done via logistic regression. Unsupervised learning, on the other hand, uses the intuition that self-harm posts will probably score high in similarity with each other, and would tend to cluster with each other. To that end, spectral clustering is used to identify the cluster. According to the paper, the accuracy of its methods outranks traditional methods, such as word-embedding, and k-means clustering.

Several questions :
- In section 3.1 (Pg 97), the paper mentions feature selection via L2-1 norm regularization. I know of L1-norm and L2-norm, but what is L2-1 norm regularization? Should I read up on it?
- In section 3.2 (Pg 98), the technique of spectral clustering seems quite technically involved. I vaguely understand the first part, up to the Laplacian Matrix (which happens to be a key idea in my Math Masters Thesis actually haha), but I'm lost after that at the Lagrangian Function part. Should I read up more on that too?