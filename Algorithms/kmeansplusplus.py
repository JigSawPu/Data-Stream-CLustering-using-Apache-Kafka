import numpy as np
import copy

class KMeans:
    '''Initialize the class with default k and max_iter.'''
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        '''This function uses k means algorithm to fit the input dataset'''
        # Set the initial centroids to randomly selected points in data
        self.centroids = self.ppinit(X)
        for i in range(self.max_iter):
            # Calculate the distance of all instances from all the cluster
            # Set label of each instance based on their distance from the nearest cluster
            distances = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
            self.labels = np.argmin(distances, axis=0)
            # For instances in same cluster, calculate their centroid and update
            thisiter = copy.deepcopy(self.centroids)
            for j in range(self.num_clusters):
                thisiter[j] = X[self.labels == j].mean(axis=0)
            if np.all(thisiter == self.centroids):
                print(f'DCDP took {self._Convergence_criteria} iterations to converge')
                break
            else:
                self.centroids = thisiter

    def ppinit(self, X): 
        initial_center = X[np.random.choice(X.shape[0], 1, replace=False)]
        for _ in range(self.k-1):
            distances = ((X - initial_center[:, np.newaxis])**2).sum(axis=2)
            max_dist_index = np.unravel_index(np.argmax(distances, axis=None), distances.shape)[1]
            next_center = X[max_dist_index]
            initial_center = np.vstack((initial_center, next_center))
        return initial_center


    def predict(self, X):
        '''This function predict the label of the instance by calculating the
        distance from the nearest cluster'''
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
