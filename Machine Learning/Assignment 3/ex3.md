Multi-class Classification - Feedforward Propagation

This exercise uses logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today — from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.

In the first part of the exercise, you will extend your previous implementation of logistic regression and apply it to one-versus-all classification. There are 5000 training examples in `ex3data1.mat`, where each training example is a 20x20 pixel grayscale image of a digit. Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.

The second part of the training set is a 5000-dimensional vector that contains labels for the training set (0 digit is labeled as 10).
