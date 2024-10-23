from DecisionTree import DecisionTree
from DataSet import DataSet
import pandas as pd

class FruitDataSet(DataSet):
    def __init__(self):
        super().__init__()
        self.data =  {'Colour': ['Green','Yellow','Red','Red','Yellow'], 
                      'Diameter': [3, 3, 1, 2, 3],
                      'Label': ['Apple','Apple','Grape','Grape','Lemon']}
        self.data_frame = pd.DataFrame(self.data)
        self.target = 'Label'
        self.features = self.data_frame.drop(self.target, axis=1).columns

class TestFruitDataSet(DataSet):
    def __init__(self):
        super().__init__()
        self.data =  {'Colour': ['Green','Yellow','Red','Red','Yellow'], 
                      'Diameter': [3, 4, 2, 1, 3],
                      'Label': ['Apple','Apple','Grape','Grape','Lemon']}
        self.data_frame = pd.DataFrame(self.data)
        self.target = 'Label'
        self.features = self.data_frame.drop(self.target, axis=1).columns


if __name__ == "__main__":
    # Create training data
    train = FruitDataSet()
    # Create tree
    tree = DecisionTree()
    # Train the tree
    tree.train(train)
    print(tree)

    # Create test data
    test = TestFruitDataSet()

    # Use the tree to predict on the data
    predictions = tree.predict(test)
    print(predictions)