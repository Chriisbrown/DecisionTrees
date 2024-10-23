import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

class DataSet:
    """
    Dataset class 
    data_frame: the pandas dataframe with the data in it, both X and y
    impurity: the gini impurity of the dataset
    target: which feature is the target of the training (y)
    features: which features are used for training (X)
    """
    def __init__(self):
        self.data_frame = None
        self.impurity = 1
        self.target = "target"
        self.features = []
        
    def find_unique(self,column : str)-> int :
        """
        Find unique values of a given feautre
        """
        return self.data_frame[column].unique()
    
    def count_class(self,label):
        """
        Find number of classes of a given label in a dataframe
        """
        return self.data_frame[self.data_frame['Label']==label].shape[0]
    
    def count_classes(self) -> dict:
        """
        Count individual instances of a given label in the dataset
        Return a dict formatted {Label : # occurances}
        """
        classes_dict = {}
        # Iterate through target labels
        for iclass in self.data_frame['Label'].unique():
            # Count how many of a given label is in the dataset
            classes_dict[iclass] = self.count_class(iclass)
        return classes_dict
    
    def gini_impurity(self) -> float:
        self.impurity = 1
        # Create classes dictionary 
        classes_count = self.count_classes()
        # Iterate through classes
        for class_label in classes_count:
            # Frequency of a given label in the total dataset
            frequency_of_label = classes_count[class_label] / len(self.data_frame)
            # Impurity calculated based on frequency squared,
            # more labels results in higher impurity
            self.impurity -= frequency_of_label**2
        return self.impurity

    def length(self) -> int:
        # length of data
        return len(self.data_frame)
    


