from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class Sampler():

    """
    problem: 2 method to sample
    1. before cleaner, oversampling minority class, but due to groupby not yet, so how to combine to new data?
    2. after cleaner and splitter, oversampling minority class (Treat each unique label combination as a "class")
        -> it seems a little problem.
    Now: function overSampling using method2.
    """
    
    splitter = None
    mlb = MultiLabelBinarizer()
    ros = RandomOverSampler(random_state=42)
    
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    X_train_resamp = None
    y_train_resamp = None

    dataFrame_label = None
    Series_label = None
    label = 'genre_list'

    def __init__(self, splitter):
        self.splitter = splitter
        if self.splitter is None:
            raise ValueError(f"splitter is None")
        self.X_train = splitter.getX_train()
        self.X_test = splitter.getX_test()
        self.y_train = splitter.gety_train()
        self.y_test = splitter.gety_test()
        print(f"training set size for sampler: {self.X_train.shape}") # (28701, 13)
        print(f"testing set size for sampler: {self.X_test.shape}") # (7176, 13)
        label_list = self.mlb.fit_transform(self.y_train[self.label])
        self.__sizeLabel(label_list, 'Before')
        self.Series_label = self.__labelCombination()
    
    def __sizeLabel(self, label_list, state):
        self.dataFrame_label = pd.DataFrame(label_list, columns=self.mlb.classes_)
        print(f"{state} Oversampling:")
        print(self.dataFrame_label.sum(axis=0))

    def __labelCombination(self):
        label_combinations = self.dataFrame_label.apply(lambda row: tuple(row), axis=1)
        # print(type(label_combinations)) # Series
        return label_combinations

    def overSampling(self):
        print(self.X_train.shape)
        self.X_train_resamp, self.y_train_resamp = self.ros.fit_resample(self.X_train, self.Series_label)
        print(self.X_train_resamp)
        print(self.X_train_resamp.shape)
        print(self.y_train_resamp)
        print(self.y_train_resamp.shape)
        