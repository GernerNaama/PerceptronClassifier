import os
import random
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.k = 0
        self.wVectors = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        self.k = len(np.unique(y))
        self.wVectors = np.zeros(shape=[self.k, X.shape[1]])        # 2D array for the W vectors

        correct_pred_flag = 0
        while correct_pred_flag == 0:       # while there are still mistakes - the correct_pred_flag is 0
            train_index = 0
            while train_index < len(X):     # going over the train set
                curr_max = float('-inf')
                y_pred = 0
                for i in range(len(self.wVectors)):         # finding argmax of inner product of X with all W vectors
                    if curr_max <= np.inner(self.wVectors[i], X[train_index]):
                        curr_max = np.inner(self.wVectors[i], X[train_index])
                        y_pred = i

                y_real = y[train_index]
                if y_pred != y_real:            # if there is prediction mistake, update the W vectors according the Perceptron Classifier
                    correct_pred_flag = 0
                    self.wVectors[y_real] += X[train_index]
                    self.wVectors[y_pred] -= X[train_index]
                    break                       # found mistake, start from the beginning (train index = 0) with the new W vectors

                else:
                    correct_pred_flag = 1       # no mistake was found

                train_index += 1


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        test_index = 0
        prediction_arr = []

        while test_index < len(X):
            curr_max = float('-inf')
            y_pred = 0
            for i in range(len(self.wVectors)):         # finding argmax of inner product of X with all W vectors
                if curr_max <= np.inner(self.wVectors[i], X[test_index]):
                    curr_max = np.inner(self.wVectors[i], X[test_index])
                    y_pred = i
            prediction_arr.append(y_pred)
            test_index += 1

        return np.array(prediction_arr)


if __name__ == "__main__":

    print("*" * 20)
    print("Started PerceptronClassifier.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
