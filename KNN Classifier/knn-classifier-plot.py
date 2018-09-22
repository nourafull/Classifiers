# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 08:26:26 2018

@author: noura
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

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

def generate_data(n):
    """generate normally distributed synthetic data"""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))),axis=0)
    """"the first half of the points belongs to class 0, the second half belongs to class 1"""
    outcomes = np.concatenate((np.repeat(0,n),np.repeat(1,n)))
    return (points,outcomes)


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


n=100
predictors, outcomes = generate_data(n)
#plt.plot(points[:n,0],points[:n,1],"ro",label="class_0") 
#plt.plot(points[n:,0],points[n:,1],"bo",label="class_1")
#plt.savefig("generated-bivariate-data.pdf")
limits = (-3,4,-3,4)
h = 0.1
k=50
filename = "knn50.pdf"

(xx,yy,prediction_grid) = prediction_grid(predictors,outcomes,limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid,filename)