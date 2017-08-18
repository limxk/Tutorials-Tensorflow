Task :
- Implement CNN on MNIST. This is the first time I've encountered CNN, so I took a bit more time to read up and understand it. Of particular note is this great lecture by another Stanford module I found on youtube : https://www.youtube.com/watch?v=LxfUGhug-iQ&t=993s

Main idea of CNN (please correct me if im wrong anywhere here) :

- CNN mainly utilises filters (kernels) that apply locally to each successive position of the image. Several (up to tens/hundreds) of different filters are used at once, and each filter focuses on a certain "property" (such as edges, sharpening, blurring, red tint etc), depending on the values of the filter

- Padding is possibly used to maintain the size of the output.

- After each filter is applied once, the output is possibly applied onto other algorithms (relu, max pooling). The resultant output is then reapplied to yet another batch of filters. this process continues several (up to hundreds) of times.

- The key of CNN is to get the dimentions of each input / output / filter right. The 4 key hyperparameters are num of filters, size of filters, stride, padding. Other hyperparameters include size of the image, depth (num of channels) of the image, batch size, etc. 

- At the end, the final output is flatenned into a 1d vector, and standard gradient descent optimization is applied to it.


Problem :

- The accuracy of the final weights is super low! At about 10%. I tried tweaking the num of epochs, the batch size, and even tried the uploaded "correct code" by the lecturer on github, but I still got the same result. Not sure where the problem is, gotta experiment more.

Question :

- One thing that bugs me is that the filters now seem to be mostly randomly initialised and trained. Since there is a physical interpretation of filters, might it be possible to initialise some filters as known kernels before trianing? (most notably would be the edge filters, since intuitively that is how humans recognise images, by first noticing the outlines of the shapes). Not sure if that will improve the optimisation/accuracy
