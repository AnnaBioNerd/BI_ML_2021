import numpy as np
import pandas as pd


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist = np.zeros((len(X),len(self.train_X)))
        for i in range(len(X)):
            for j in range(len(self.train_X)):
                dist[i,j] = np.sum(abs(np.subtract(self.train_X[j],X[i])))
        return dist
        
        """
        YOUR CODE IS HERE
        """
        pass


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist = np.zeros((len(X),len(self.train_X)))
        for i in range(len(X)):
            d = self.train_X - X[i]
            dist[i] = abs(d).sum(axis = 1)
        return dist
        """
        YOUR CODE IS HERE
        """
        
        pass


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        d = self.train_X - X[:,np.newaxis]
        d = abs(d)
        dist = d.sum(axis = 2)
        return dist
        """
        YOUR CODE IS HERE
        """
        pass


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        for i in range(n_test):
            df1 = pd.DataFrame({'x':distances[i],'y':self.train_y})
            df1 = df1.sort_values(by =['x'])
            df1 = df1['y'][0:self.k]
            df1 = df1.mode()
            prediction[i] = df1[0]
        prediction = list(prediction)
        prediction = [str(int(i)) for i in prediction]
        return prediction
        """
        YOUR CODE IS HERE
        """
        pass


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        for i in range(n_test):
            df1 = pd.DataFrame({'x':distances[i],'y':self.train_y})
            df1 = df1.sort_values(by =['x'])
            df1 = df1['y'][0:self.k]
            df1 = df1.mode()
            prediction[i] = df1[0]
        prediction = list(prediction)
        prediction = [str(int(i)) for i in prediction]
        return prediction
        """
        YOUR CODE IS HERE
        """
        pass