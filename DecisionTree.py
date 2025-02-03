from DataSet import DataSet
from Split import *

class Node:
    """
    Node of Decision Tree
    Contains the Split at the node
    Contains reference to the left and right children of the node
    Which are either nodes or leaves
    """
    def __init__(self, split : Split,child_left,child_right):
        self.split = split
        self.child_branch_left = child_left
        self.child_branch_right = child_right

class Leaf:
    """"
    Leaf of Decision Tree
    Returns the prediction on a given data split
    """
    def __init__(self,data : DataSet) -> dict:
        self.predictions = data.count_classes()

class DecisionTree():
    """
    Decision Tree class
    Contains the user functions "train, predict" and internal functions
    Depth of the tree is controlled by the max depth parameter
    """
    def __init__(self, max_depth : int = 1):
        self.max_depth = max_depth
        self.tree = None

    def _build_tree(self,data : DataSet, current_depth : int = 0) -> None:
        """
        Recursive function for building the tree, once a split has been created
        data is partitioned and the function is rerun on the partitioned data
        until max depth has been reached or there is no information gain at a node
        """
        # split the data to give the split that produces the highest gain in information
        gain, split = find_best_split(data, BinaryAxisAlignedSplit)
        # If gain is 0 or the tree is too deep return a leaf node and stop the recursion
        if gain == 0 or current_depth > self.max_depth:
            return Leaf(data)

        # Partition the dataset into left and right datasets based on the split
        child_dataset_left, child_dataset_right = split.partition(data)
        # Recurse the function, run again on the split dataset with two new trees
        child_branch_left = self._build_tree(child_dataset_left,current_depth)
        child_branch_right = self._build_tree(child_dataset_right,current_depth)
        current_depth += 1
        # Return a node with its split and the references to the left and right children
        return Node(split, child_branch_left,child_branch_right)

    def train(self,data : DataSet) -> None:
        """
        Wrapper function for training, builds the tree and stores it internally 
        """
        self.tree = self._build_tree(data)

    def _classify(self,data : DataSet, node) -> None:
        """
        Recursive function for traversing the tree node by node until
        reaching the leaves and the predictions
        """
        if isinstance(node, Leaf):
            return node.predictions
        
        # `Recursively go left or right at the node split`
        if node.split.compare(data):
            return self._classify(data, node.child_branch_left)
        else:
            return self._classify(data, node.child_branch_right)
        
    def predict(self,data : DataSet) -> list:
        """
        Wrapper function for predicting, classifies a dataset row by row storing 
        in a predicitons list
        """
        predictions = []
        for index, row in data.data_frame.iterrows():
            predictions.append(self._classify(row,self.tree))
        return predictions
    
    def _print_tree(self,node, spacing="") -> None:
        """
        Print the tree, recursively
        """

        # Print a leaf
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the split at a node
        print (spacing + str(node.split))

        # Call this function recursively on the left branch adding 
        # a space for pretty printing
        print (spacing + '--> Left:')
        self._print_tree(node.child_branch_left, spacing + "  ")

        # Call this function recursively on the right branch
        print (spacing + '--> Right:')
        self._print_tree(node.child_branch_right, spacing + "  ")

    def __repr__(self) -> str:
        """
        Wrapper function around the print tree, called by print(tree_object)
        """
        self._print_tree(self.tree)
        return ''
