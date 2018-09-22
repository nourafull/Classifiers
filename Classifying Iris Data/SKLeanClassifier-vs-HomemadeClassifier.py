# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:17:04 2018
@author: noura
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import random

def distance(p1,p2):
    """calculate the euclidean distance between two points"""
    return np.sqrt(np.sum(np.power(p2-p1,2)))

def majority_vote(votes):
    """"Return the most common element in the list votes"""
    vote_count = {}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote] += 1
        else:
            vote_count[vote] = 1
    """collect all ties in the list winners"""
    winners = []
    max_count = max(vote_count.values())
    for vote, count in vote_count.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners)

def find_nearest_neighbours(p,points,k=3):
    """finding the k nearest neighbours to the point p"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p,points[i])   
    ind = np.argsort(distances)
    #print(distances[ind])
    return ind[:k]

def knn_prediction(p,points,outcomes,k=3):
    """K-nearest neighbour classifier"""
    ind = find_nearest_neighbours(p,points,k)
    return majority_vote(outcomes[ind])


"""-----------Load the Iris dataset, then extract predictors and outcomes-----------"""
iris = datasets.load_iris()
predictors = iris.data[:,0:2]
outcomes = iris.target

"""  -----------Classify Iris Data using our homemade KNN Classifier -----------"""
homemade_predictions = np.array([knn_prediction(p,predictors,outcomes,5) for p in predictors])

"""  -----------Classify Iris Data using SKLearn's KNN Classifier -----------"""
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors,outcomes)
sk_predictions = knn.predict(predictors)

"""  -----------SKLearn's KNN Classifier vs homemade KNN Classifier -----------"""
sklearn_vs_homemade = 100 * np.mean(homemade_predictions == sk_predictions)
homemade_accuracy = 100 * np.mean(homemade_predictions == outcomes)
sklearn_accuracy = 100 * np.mean(sk_predictions == outcomes)

print("the scklearn classifier agrees with our homemade classifier ",sklearn_vs_homemade,"% of the time")
print("the homemade classifier has ",homemade_accuracy,"% accuracy")
print("the sklearn classifier has ",sklearn_accuracy,"% accuracy")