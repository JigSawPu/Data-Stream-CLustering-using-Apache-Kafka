import copy
import pandas as pd
from collections import deque

# a non recursive version of coreset tree 
# that use stacks and queues for faster access

class dfNode:
    def __init__(self, subset, q):
        self.subset = subset
        self.q = q
        self.cost = self.distCalc(self.subset, self.q).sum()
        self.weight = self.subset.shape[0]

        self.leftChild = None
        self.rightChild = None
        self.parent = None

    def distCalc(self, miniset, x):
        """
        distCalc Calculate distance of each point in miniset to x

        Args:
            miniset (_type_): a subset
            x (_type_): representative point

        Returns:
            _type_: sum of distances
        """
        distances = (miniset.subtract(x.values)**2).sum(axis=1)
        return distances

    def subsetFinder(self):
        """
        subsetFinder split the parent node into two children based on
        the cost of splitting
        Returns:
            _type_: tuple of subsets and its representative points
        """
        dist = self.distCalc(self.subset, self.q)
        # max_dist_index = np.unravel_index(np.argmax(dist, axis=None),
        #                                   dist.shape)[1]
        max_dist_index = dist.idxmax() #change3
        q2 = self.subset.loc[max_dist_index]
        q2 = q2.to_frame().transpose()
        q2dist = self.distCalc(self.subset, q2)
        group_dist = pd.concat([dist, q2dist], axis=1)
        nearest_indices = group_dist.idxmin(axis=1)
        subset1 = self.subset[nearest_indices == 0]
        subset2 = self.subset[nearest_indices == 1]
        return (subset1, self.q), (subset2, q2)


class CoreSetTree:
    def __init__(self, X, m):
        """
        __init__ initialization

        Args:
            X (pandas dataframe): pass the dataframe from 
            which to calculate the core set tree
            m (_type_): number of points to have in the set tree
        """
        self.wholeSet = X
        self.m = m
        self.coreset = pd.DataFrame([])
        self.iter = 0
        self.all_leaves = deque()
    def fit(self):
        q1 = self.wholeSet.sample()
        self.Root = dfNode(self.wholeSet, q1)
        for i in range(self.m):
            self.maxnode = None
            self.maxcost = 0
            self.vist_add_propagate(self.Root)
        for node in self.all_leaves:
            if type(node.leftChild) is type(None) and type(node.rightChild) is type(None):
                self.coreset = pd.concat([self.coreset, node.q])
    def vist_add_propagate(self, root):
        q = deque()
        q.append(root)
        ans = deque()
        while q:
            node = q.popleft()
            if node is None:
                continue
            ans.appendleft(node)
            if node.rightChild:
                q.append(node.rightChild)
            if node.leftChild:
                q.append(node.leftChild)
        for node in ans:
            if type(node.leftChild) is type(None) and type(node.rightChild) is type(None):
                if node.cost >= self.maxcost:
                    self.maxcost = node.cost
                    self.maxnode = node
        childs = self.maxnode.subsetFinder()
        for node in ans:
            if node == self.maxnode:
                node.leftChild = dfNode(childs[0][0],
                                                    childs[0][1])
                node.rightChild = dfNode(childs[1][0],
                                        childs[1][1])
                node.rightChild.parent = node
                node.leftChild.parent = node
        ans.appendleft(node.rightChild)
        ans.appendleft(node.leftChild)
        ans.pop()
        self.iter += 1
        if self.iter == self.m:
            self.all_leaves = copy.deepcopy(ans)
        while ans:
            left = ans.popleft()
            right = ans.popleft()
            left.parent.cost = left.cost + right.cost
