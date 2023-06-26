import pandas as pd

# Node class, 
#

class dfNode:
    """_summary_
    """
    def __init__(self, subset, q): 
        """_summary_
        """       
        self.subset = subset
        self.q = q
        self.cost = self.distCalc(self.subset, self.q).sum()
        self.weight = self.subset.shape[0]
        
        self.leftChild = None
        self.rightChild = None
    
    def distCalc(self, miniset, x):
        """
        distCalc Calculate distance of each point in miniset to x

        Args:
            miniset (dataframe): a subset
            x: representative point

        Returns:
            : sum of distances
        """
        distances = (miniset.subtract(x.values)**2).sum(axis=1)
        return distances

    def subsetFinder(self):
        """
        subsetFinder split the parent node into two children based on
        the cost of splitting
        Returns:
            : tuple of subsets and its representative points
        """
        dist = self.distCalc(self.subset, self.q)
        max_dist_index = dist.idxmax()
        q2 = self.subset.loc[max_dist_index]
        q2 = q2.to_frame().transpose()
        q2dist = self.distCalc(self.subset, q2)
        group_dist = pd.concat([dist, q2dist], axis=1)
        nearest_indices = group_dist.idxmin(axis=1)
        subset1 = self.subset[nearest_indices == 0]
        subset2 = self.subset[nearest_indices == 1]
        return (subset1,self.q),(subset2,q2)
    


class dfCoreSetTree:
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
        self.coreset = pd.DataFrame([], columns=['x','y'])

    def fit(self):
        q1 = self.wholeSet.sample()
        self.coreset = pd.concat([self.coreset, q1])
        self.Root = dfNode(self.wholeSet, q1) 
        self.MaxNode = None
        self.MaxNodeCost = 0
        for i in range(self.m):
            self.MaxNode = None
            self.MaxNodeCost = 0
            self.visit(self.Root)
            self.addChild(self.Root)
            self.propagateUp(self.Root)
    
    def addChild(self, node):
        if node:
            self.addChild(node.leftChild)
            self.addChild(node.rightChild)
            if type(node.leftChild) is type(None) and type(node.rightChild) is type(None):
                if node.cost == self.MaxNodeCost:
                    childs = self.MaxNode.subsetFinder()
                    self.coreset = pd.concat([self.coreset, childs[1][1]])
                    node.leftChild = dfNode(childs[0][0], 
                                          childs[0][1])
                    node.rightChild = dfNode(childs[1][0],
                                           childs[1][1])
    
    def visit(self, node):
        if node:
            self.visit(node.leftChild)
            self.visit(node.rightChild)
            if type(node.leftChild) is type(None) and type(node.rightChild) is type(None):
                if node.cost >= self.MaxNodeCost:
                    self.MaxNode = node
                    self.MaxNodeCost = node.cost
    
    def propagateUp(self, node):
        if node:
            self.propagateUp(node.leftChild)
            self.propagateUp(node.rightChild)
            if node.leftChild and node.rightChild:
                node.cost = node.leftChild.cost + node.rightChild.cost
