import os
import pandas as pd

from checker import Checker
from cleaner import Cleaner
from splitter import Splitter
from preprocessing import Preprocessing
from classifier import Classifier
from evaluator import Evaluator
from cluster import Cluster

"""
explanation of features from spofify API documentation
https://developer.spotify.com/documentation/web-api/reference/get-audio-features
"""

def read_csv(csvFile):
    """
    Read the csv file and return the dataframe
    args:
        --.csv file: csv file
    return:
        --dataframe: dataframe
    """
    try:
        df = pd.read_csv(csvFile, dtype={   
                                            "song_name": "str", 
                                            "key": "category",
                                            "mode": "category",
                                            "time_signature": "category"
                                        })
    except Exception as e:
        raise e
    return df

def main():

    """##############################################"""
    """
    initial setup, lncluding reading and check dataframe with differnet columns
    """
    dirPath = "./data/"
    fileName = "genres_v2.csv"
    csvFile = os.path.join(dirPath, fileName)
    df = read_csv(csvFile)
    # print(df.head())
    """##############################################"""

    """##############################################"""
    """
    remove the columns that are not needed
    """
    drop_columns = ["uri", "track_href", "analysis_url", "type", "song_name", "Unnamed: 0", "title"]
    df.drop(drop_columns, axis=1, inplace=True)
    """##############################################"""

    """##############################################"""
    """
    create an instance of the class checker to check the columns
    """
    checker = Checker(df)
    checker.describeStatistic()
    checker.checkMissing()
    outliers = checker.checkNoise()
    # print(outliers)
    """##############################################"""

    """##############################################"""
    """
    data cleaning, including groupby and multi-label encoding
    """
    cleaner = Cleaner(df, outliers) # size of cleaner.df = 35877 rows
    df_cleaned = cleaner.getDataFrame()
    """##############################################"""

    """##############################################"""
    """
    data splitting
    """
    splitter = Splitter(df_cleaned)
    X_train = splitter.getX_train()
    X_test = splitter.getX_test()
    y_train = splitter.gety_train()
    y_test = splitter.gety_test()
    trainingSet = splitter.getTrainingSet()
    validationSet = splitter.getValidationSet()
    """##############################################"""

    """##############################################"""
    """
    data preprocessing, including normalization and one-hot encoding with preprocessor
    """
    preprocessing = Preprocessing()
    preprocessor = preprocessing.getPreprocessor()
    """##############################################"""

    """##############################################"""
    """
    classification
    """
    classifier = Classifier(preprocessor, X_train, y_train)
    classifier.randomForest()
    classifier.SVM()
    classifier.linearSVM()
    pipeline_rf = classifier.getPipeline('RandomForest')
    pipeline_SVM = classifier.getPipeline('SVM')
    pipeline_linearSVM = classifier.getPipeline('LinearSVM')
    # classifier.getFeatureImportance(pipeline_rf, 'RandomForest')
    # classifier.getFeatureImportance(pipeline_linearSVM, 'LinearSVM')
    """##############################################"""

    """##############################################"""
    """
    evaluation
    """
    evaluator = Evaluator(X_test, y_test)
    # evaluator.evaluate(pipeline_rf, 'RandomForest')
    # evaluator.evaluate(pipeline_SVM, 'SVM')
    # evaluator.evaluate(pipeline_linearSVM, 'LinearSVM')
    # evaluator.permutationImportance(pipeline_rf, 'RandomForest')
    # evaluator.permutationImportance(pipeline_SVM, 'SVM')
    # evaluator.permutationImportance(pipeline_linearSVM, 'LinearSVM')
    """##############################################"""

    """##############################################"""
    """
    clustering and evaluation
    """
    cluster = Cluster(preprocessor, trainingSet, validationSet, df=None, genreList=None)
    # cluster.clustering("Kmeans")
    # cluster.clustering("Agglomerative")
    # cluster.clustering("DBSCAN")
    # cluster.clustering("GMM")
    bestParam_Kmeans = 20
    bestParam_Agglomerative = 20
    bestParam_DBSCAN = 1.6
    bestParam_GMM = 20
    # cluster.bestCluster("Kmeans", bestParam_Kmeans)
    # cluster.bestCluster("Agglomerative", bestParam_Agglomerative)
    # cluster.bestCluster("DBSCAN", bestParam_DBSCAN)
    # cluster.bestCluster("GMM", bestParam_GMM)
    """##############################################"""

    """##############################################"""
    """
    feature selection and clustering
    """
    columns_selected_rf = ["tempo", "duration_ms", "instrumentalness", "danceability", "energy", "loudness", "acousticness", "speechiness"]
    columns_selected_SVM = ["tempo", "duration_ms", "key", "danceability", "energy", "acousticness", "loudness", "instrumentalness"]
    columns_selected_linearSVM = ["duration_ms", "tempo", "instrumentalness", "danceability", "loudness"]
    columns_selected = ["tempo", "duration_ms", "key", "danceability", "energy", "instrumentalness", "loudness"]
    columns_nominal_selected = ['key']
    columns_numeric_selected = ['danceability', 'energy', 'loudness', 'instrumentalness', 'tempo', 'duration_ms']
    columns_drop = ["time_signature", "mode", "liveness", "valence", "speechiness", "acousticness"]
    cluster.featureSelection(columns_drop)
    preprocessing.setPreprocessor(columns_numeric_selected, columns_nominal_selected)
    preprocessor = preprocessing.getPreprocessor()
    cluster.setPreprocessor(preprocessor)
    # cluster.clustering("Kmeans")
    # cluster.clustering("Agglomerative")
    # cluster.clustering("DBSCAN")
    # cluster.clustering("GMM")
    optimalList = [27, 15, 1.7, 17]
    compareList = [20, 20, 1.6, 20]
    key = "compare" # "optimal" | "compare"
    paraList = compareList if key == "compare" else optimalList
    bestParam_Kmeans = paraList[0]
    bestParam_Agglomerative = paraList[1]
    bestParam_DBSCAN = paraList[2]
    bestParam_GMM = paraList[3]
    cluster.bestCluster("Kmeans", bestParam_Kmeans)
    cluster.bestCluster("Agglomerative", bestParam_Agglomerative)
    cluster.bestCluster("DBSCAN", bestParam_DBSCAN)
    cluster.bestCluster("GMM", bestParam_GMM)
    """##############################################"""
if __name__ == "__main__":
    main()