from sklearn.model_selection import train_test_split

class Splitter:

    df = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    columns_nominal = ['key', 'mode', 'time_signature']
    columns_numeric = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    columns_label = ['genre_list']

    def __init__(self, df):
        self.df = df
        if self.df is None:
            raise ValueError("df is None")
        else:
            self.__split()

    def __split(self):
        """
        split the dataframe into training and testing set
        """
        X = self.df[self.columns_nominal + self.columns_numeric]
        y = self.df[self.columns_label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"training set size: {self.X_train.shape}") # (28701, 13)
        print(f"testing set size: {self.X_test.shape}") # (7176, 13)

    def getX_train(self):
        return self.X_train
    
    def getX_test(self):
        return self.X_test
    
    def gety_train(self):
        return self.y_train
    
    def gety_test(self):
        return self.y_test
    
    def getTrainingSet(self):
        return self.df[self.columns_nominal + self.columns_numeric]
    
    def getValidationSet(self):
        return self.df[self.columns_label]