from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Preprocessing:

    preprocessor = None
    columns_nominal = ['key', 'mode', 'time_signature']
    columns_numeric = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    scaler = RobustScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def __init__(self):
        self.__preprocess()

    def __preprocess(self):
        """
        preprocess the dataframe including normalization and one-hot encoding
        args:
            --None
        return:
            --None
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.scaler, self.columns_numeric),
                ('cat', self.encoder, self.columns_nominal)
            ]
        )

    def getPreprocessor(self):
        return self.preprocessor
    
    def setPreprocessor(self, columns_numeric_selected, columns_nominal_selected):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.scaler, columns_numeric_selected),
                ('cat', self.encoder, columns_nominal_selected)
            ]
        )
        self.preprocessor = preprocessor