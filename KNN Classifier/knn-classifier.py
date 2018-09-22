import numpy as np
import matplotlib.pyplot as plt
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


points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p = np.array([2,1.9])
outcomes = np.array([0,0,0,0,1,1,1,1,1])

prediction = knn_prediction(p,points,outcomes,5)
print("Point ",p," is classified in class ",prediction)

#plt.plot(points[:,0],points[:,1],"ro")
#plt.plot(p[0],p[1],"bo")
#plt.axis([0.5,3.5,0.5,3.5])

plt.plot(points[:4,0],points[:4,1],"go",label="class_0") 
plt.plot(points[4:,0],points[4:,1],"bo",label="class_1")
plt.plot(p[0],p[1],"rx",label="new observation")
plt.axis([0.5,4.5,0.5,4.5])
plt.legend()