import collections
import math
import numpy as np


class Gaussian_Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """
        self.x = X_train
        self.y = y_train

        self.gen_by_class()

    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = None

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)
        classes = np.unique(self.y)
        counts = dict()
        for c in classes:
            self.x_by_class[c] = []
            self.mean_by_class[c] = []
            self.std_by_class[c] = []
            counts[c] = 0
        for i in range(0, self.x.shape[0]):
            c = self.y[i]
            self.x_by_class[c].append(self.x[i])
            counts[c] = counts[c] + 1
        for c in classes:
            self.mean_by_class[c] = self.mean(self.x_by_class[c])
            self.std_by_class[c] = self.std(self.x_by_class[c])
        self.y_prior = np.array([(counts[c] / len(self.y)) for c in classes])
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        total = 0
        for i in x:
            total += i
        mean = total / len(x)
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean

    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std
        mean = self.mean(x)
        var = sum([(n-mean)**2 for n in x]) / (len(x)-1)
        std = [math.sqrt(n) for n in var]
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return np.array(std)

    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std
        for i in range(len(std)):
            if std[i] == 0:
                std[i] = 1e-5
        dist = np.exp(-((x-mean)**2 / (2*std**2))) / (math.sqrt(2*math.pi)*std)
        for i in range(len(dist)):
            if dist[i] == 0:
                dist[i] = 1e-5
        return dist
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """

        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        for arr in range(n):
            for c in range(num_class):
                prediction[arr][c] = self.y_prior[c] + np.sum(np.log(self.calc_gaussian_dist(x[arr], self.mean_by_class[c], self.std_by_class[c])))
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

        return prediction


class Neural_Network():
    def __init__(self, hidden_size=64, output_size=1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.hidden_size = hidden_size
        self.output_size = output_size

    def fit(self, x, y, batch_size=64, iteration=2000, learning_rate=1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """
        dim = x.shape[1]
        num_train = x.shape[0]

        # initialize W
        if self.W1 is None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0

            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0

        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            loss, gradient = self.loss(x_batch, y_batch)

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method
            self.W1 = self.W1 - learning_rate * gradient["dW1"]
            self.W2 = self.W2 - learning_rate * gradient["dW2"]
            self.b1 = self.b1 - learning_rate * gradient["db1"]
            self.b2 = self.b2 - learning_rate * gradient["db2"]
            pass
            # END_YOUR_CODE
            ############################################################
            ############################################################

            y_pred = self.predict(x_batch)
            acc = np.mean(y_pred == y_batch)

            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def loss(self, x_batch, y_batch, reg=1e-3):
        """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives. 

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer.

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
        gradient = {'dW1': None, 'db1': None, 'dW2': None, 'db2': None}

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate y_hat which is probability of the instance is y = 0.
        g1 = x_batch.dot(self.W1) + self.b1
        h1 = self.activation(g1)
        g2 = h1.dot(self.W2) + self.b2
        y_hat = self.sigmoid(g2)
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate loss and gradient
        n = y_hat.shape[0]
        loss = -np.sum(y_batch * np.log(y_hat + 1e-5)) / n

        gradient['dW1'] = x_batch.transpose().dot(y_hat-y_batch)*self.W2.transpose() / (2*n)
        gradient['db1'] = np.mean((y_hat-y_batch)*self.W2) / n
        gradient['dW2'] = h1.transpose().dot(y_hat-y_batch) / (2*n)
        gradient['db2'] = np.mean(y_hat-y_batch) / n
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU 
        s = np.maximum(0, z)
        pass

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        s = 1 / (1 + np.exp(-z))
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y
        g1 = x.dot(self.W1) + self.b1
        h1 = self.activation(g1)
        g2 = h1.dot(self.W2) + self.b2
        y_hat = np.round(self.sigmoid(g2))
        pass

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_hat
