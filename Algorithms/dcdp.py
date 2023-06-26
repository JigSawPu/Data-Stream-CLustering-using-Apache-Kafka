import copy
import csv
import numpy as np
from loguru import logger

class DC_DP:
    def __init__(self, lmbda, max_iter: int):
        '''Initialize the class with default lambda and convergence criteria.'''
        self.lmbda = lmbda
        self._Convergence_criteria = 0
        self.max_iter = max_iter

    def fit(self, X):
        # Set the initial no of clusters to 1
        # Set Centroid to mean of input data
        # Set defaults labels to 0
        self.num_clusters = 1
        self.centroids = X.mean(axis=0)[np.newaxis,:]
        self.labels = np.zeros(len(X))
        # Use while loop to control convergence
        # Use i_max to save the index of instance whose distance from nearest
        # cluster is greater than lambda in one iteration
        # Use d_max to save the distance of that instance
        while self._Convergence_criteria < self.max_iter:
            i_max = -1
            d_max = -1
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            # If the distance of instance from nearest cluster is greater than lambda
            # Set i_max to d, set d_max to distane from nearest cluster
            for d in range(len(distances.T)):
                if distances[np.argmin(distances[...,d]),d] > d_max:
                    i_max = d
                    d_max = distances[np.argmin(distances[...,d]),d]
            # After assignment step is complete, if there is one instance
            # whose d_max is greater than lambda, create a new cluster
            # Initialize the cluster with that instance as the new centroid
            # of that cluster
            # Set the label of cluster as (num cluster - 1)
            if d_max > self.lmbda:
                self.num_clusters += 1
                self.centroids = np.vstack((self.centroids, X[i_max]))
                self.labels[i_max] = self.num_clusters - 1
            # For instances in same cluster, calculate their centroid and update
            thisiter = copy.deepcopy(self.centroids)
            for j in range(self.num_clusters):
                thisiter[j] = X[self.labels == j].mean(axis=0)
            if np.all(thisiter == self.centroids):
                logger.info(f'DCDP took {self._Convergence_criteria} iterations to converge')
                with open('./Iterations_to_converge.csv','a') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([f"Algorithm took Iterations to converge: ", self._Convergence_criteria])
                break
            else:
                self.centroids = thisiter
            self._Convergence_criteria += 1
        logger.info(f'DCDP took {self._Convergence_criteria} iterations to converge')
        with open('./Iterations_to_converge.csv','a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([f"Algorithm took Iterations to converge: ", self._Convergence_criteria])
        
        
    def predict(self, X):
        '''This function predict the label of the instance by calculating the
        distance from the nearest cluster'''
        distances = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
        return np.argmin(distances, axis=0)