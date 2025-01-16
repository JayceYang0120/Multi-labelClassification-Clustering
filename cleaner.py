

class Cleaner:

    df = None
    outliers = None
    columns_nominal = ['key', 'mode', 'time_signature']
    columns_numeric = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

    def __init__(self, df, outliers):
        self.df = df
        self.outliers = outliers
        if not self.df.empty:
            self.__build()
    
    def __build(self):
        # self.__removeOutliers() # ignore the outliers first, handle it later
        self.__groupbyID() # 42305 -> 35877 rows

    def __groupbyID(self):
        """
        group by id and aggregate the genre into a list named genre_list
        args:
            --None
        return:
            --None
        """
        df = self.df.copy(deep=True)
        genre_list = df.groupby('id')['genre'].apply(list).reset_index()
        df = df.merge(genre_list, on='id', suffixes=('', '_list'))
        df['genre_list'] = df['genre_list'].apply(lambda x: list(set(x)) if isinstance(x, list) else eval(str(x)))
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        df.drop(['genre', 'id'], axis=1, inplace=True)
        self.df = df.copy(deep=True)
        # print(self.df)
        ########################################
        """
        test region, for groupby function
        """
        # df = self.df.groupby("id").agg({
        #     'danceability': 'mean',
        #     'energy': 'mean',
        #     'key': 'mode',
        #     'loudness': 'mean',
        #     'mode': 'mode',
        #     'speechiness': 'mean',
        #     'acousticness': 'mean',
        #     'instrumentalness': 'mean',
        #     'liveness': 'mean',
        #     'valence': 'mean',
        #     'tempo': 'mean',
        #     'duration_ms': 'mean',
        #     'time_signature': 'mode',
        #     'genre': 'mode',
        #     'song_name': 'mode',
        #     'Unnamed: 0': 'mode',
        #     'title': 'mode',
        #     'genre_list': 'first'
        # })
        ########################################
        
    def __removeOutliers(self):
        """
        remove the outliers in the dataframe
        args:
            --None
        return:
            --None
        """
        df = self.df[~self.outliers].reset_index(drop=True)
        self.df = df.copy(deep=True)
    
    def getDataFrame(self):
        """
        get the dataframe
        args:
            --None
        return:
            --df: dataframe after groupby and multi-label encoding
        """
        return self.df