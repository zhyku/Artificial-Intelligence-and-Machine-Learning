For this exercise, the backpropagation algorithm is implemented to learn the parameters for the neural network for the last exercise's feedforward propagation algorithm to predict handwritten digits with the weights provided. The dataset is the same as that used in the previous exercise.

In the first part of the exercise, you will extend your previous implementation of logistic regression and apply it to one-versus-all classification. There are 5000 training examples in ex4data1.mat, where each training example is a 20x20 pixel grayscale image of a digit. Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.

The second part of the training set is a 5000-dimensional vector that contains labels for the training set (0 digit is labeled as 10).
