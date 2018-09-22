# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:43:03 2018

@author: noura
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def prediction_grid(predictors,outcomes,limits,h,k):
    
    (xmin,xmax,ymin,ymax) = limits
    xs = np.arange(xmin,xmax,h)
    ys = np.arange(ymin,ymax,h)
    xx,yy = np.meshgrid(xs,ys)
    
    prediction_grid = np.zeros(xx.shape,dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_prediction(p,predictors,outcomes,k)
    
    return (xx,yy,prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

"""-----------Load the Iris dataset, then extract predictors and outcomes-----------"""
iris = datasets.load_iris()
predictors = iris.data[:,0:2]
outcomes = iris.target

"""Create SKLearn KNN classifier"""
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors,outcomes)
sk_predictions = knn.predict(predictors)

"""------------Plot Predictions------------"""
limits = (4,8,1.5,4.5)
h = 0.1
k=5
filename = "IrisPredictionGrid-SKLearn-KNN-Classifier.pdf"
(xx,yy,prediction_grid) = prediction_grid(predictors,outcomes,limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid,filename)



