from DataSet import DataSet
import pandas as pd

class Split:
    """
    Super class for splitting the dataset
    Requites a feature or feautres to split on and a threshold to split with
    """
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

class BinaryAxisAlignedSplit(Split):
    """
    Binary split based on single feature and a single threshold
    Works for both numeric and categorical features
    """
    def __init__(self, feature : str, threshold : any):
        super().__init__(feature, threshold)

    def compare(self, dataframe : pd.DataFrame) -> bool:
        """
        Compare the data to the threshold, return true or fale if passes the threshold
        Compatable with both single rows of a pandas dataframe or entire dataframes 
        """
        example = dataframe[self.feature]  
        # Check if example is numeric, if numeric >= threshold
        if is_numeric(example):
            return example >= self.threshold
        # If not numeric, is categorical, return if example is the category
        else:
            return example == self.threshold
    
    def partition(self, dataset):
        """
        Partition a dataset based on the split
        """

        # Create two blank dataset objects ready for split
        dataframe_split_left = DataSet()
        dataframe_split_right = DataSet()

        # Copy features and target
        dataframe_split_left.features = dataset.features
        dataframe_split_right.features = dataset.features
        dataframe_split_left.target = dataset.target
        dataframe_split_right.target = dataset.target

        # Split dataframe based on comparison, left for passing comparison, right for failing comparison
        dataframe_split_left.data_frame = dataset.data_frame[self.compare(dataset.data_frame)]
        dataframe_split_right.data_frame = dataset.data_frame[~self.compare(dataset.data_frame)]

        return dataframe_split_left, dataframe_split_right
    
    def __repr__(self):
        """
        Print the feature and threshold
        """
        return "feature: %s, threshold %s " %(self.feature,self.threshold)


def is_numeric(x : any) -> bool:
    # Checks the input is integer or float
    return isinstance(x, int) or isinstance(x, float)

def information_gain(children : list, current_uncertainty : float):
    # Calculate the information gained from the split
    # Calculate the length of the children
    lengths = [child.length() for child in children]
    # Calculate the weighting of the children based on their length compared to
    # the size of the other children
    weights = [child.length() / sum(lengths) for child in children]
    # Uncertainty of a given child is calcualted based on their weighted gini impurity
    uncertainties = [child.gini_impurity() * weights[i] for i,child in enumerate(children) ]
    # Return total information gained from a split
    return current_uncertainty - sum(uncertainties)

def find_best_split(dataset : DataSet, SplitFunc) -> tuple:
    # Find the split that maximises the information gain, brute force check every feature, every value
    best_information_gain = 0
    best_threshold = None
    best_split = None
    # Calculate original impurity of the dataset
    dataset.gini_impurity()
    print(dataset.impurity)
    # Iterate through features
    for column in dataset.features:
        values = dataset.find_unique(column)
        # Iterate through unique values in the column
        # No point checking a feature with the same value
        for value in values:
            # Split the data based on the current feature
            # and current value in the feature
            Split = SplitFunc(column,value)
            # Partition the dataset based on the current split
            data1,data2 = Split.partition(dataset)
            # Break if there isn't anything left in one of the datasets
            if data1.length() == 0 or data2.length() == 0:
                continue
            # Calculate the gain given the current split
            gain = information_gain([data1,data2], dataset.impurity)
            if gain >= best_information_gain:
                # If this gain is the best store it
                best_information_gain, best_split = gain,Split
    # Return a tuple of the best gain and the split that caused it
    return best_information_gain, best_split


