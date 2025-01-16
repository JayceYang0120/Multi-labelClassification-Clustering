from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics.cluster import rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from sklearn.pipeline import Pipeline
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os

from visualizer import Visualizer

class Cluster():

    mlb = MultiLabelBinarizer()

    preprocessor = None
    trainingSet = None
    validationSet = None
    pipeline = None

    y_true = None

    visualizer = None

    k_clusters = range(2, 31)
    eps = [0.75 + 0.0675 * i for i in range(15)]
    
    sse = []
    silhouetteScores = []
    randScores = []
    normalizedMIScores = []
    adjustedMIScores = []
    vMeasureScores = []
    FowlkesMallowsScores = []

    def __init__(self, preprocessor, trainingSet, validationSet, df, genreList):
        self.preprocessor = preprocessor
        self.trainingSet = trainingSet
        self.validationSet = validationSet
        if self.trainingSet is None or self.validationSet is None:
            raise ValueError(f"Training set or validation set is empty")
        else:
            self.__labelEncoding()
    
    def __labelEncoding(self):
        """
        label encoding for y_true
        args:
            --None
        return:
            --None
        """
        label_list = self.mlb.fit_transform(self.validationSet.values.ravel())
        self.y_true = label_list

    def __getMaxClass(self, labels):
        """
        get the maximum class in each cluster
        args:
            --labels: list of cluster labels
            --numOfClusters: number of clusters
        return:
            --clusterClasses: dictionary of cluster classes with maximum count of each cluster
        """
        clusterClasses = {}
        uniqueLabels = set(labels)
        groundTruth = self.validationSet.values.ravel()
        for cluster in uniqueLabels:
            clusterIndices = [i for i, label in enumerate(labels) if label == cluster]
            classes_in_cluster = [cls for idx in clusterIndices for cls in groundTruth[idx]]
            class_counts = Counter(classes_in_cluster)
            representative_class = class_counts.most_common(1)[0][0]
            clusterClasses[cluster] = representative_class
        return clusterClasses
    
    def __clean(self):
        self.sse = []
        self.silhouetteScores = []
        self.randScores = []
        self.normalizedMIScores = []
        self.adjustedMIScores = []
        self.vMeasureScores = []
        self.FowlkesMallowsScores = []
    
    def __evaluate(self, y_true, y_pred):
        randScore = rand_score(y_true, y_pred)
        self.randScores.append(randScore)
        nMIScore = normalized_mutual_info_score(y_true, y_pred)
        self.normalizedMIScores.append(nMIScore)
        aMIScore = adjusted_mutual_info_score(y_true, y_pred)
        self.adjustedMIScores.append(aMIScore)
        vMeasurScore = v_measure_score(y_true, y_pred)
        self.vMeasureScores.append(vMeasurScore)
        FMScore= fowlkes_mallows_score(y_true, y_pred)
        self.FowlkesMallowsScores.append(FMScore)

    def __visualize(self, method, dirPath):
        self.visualizer = Visualizer(self.k_clusters, self.eps, method, dirPath)
        self.visualizer.visualization_Elbow(self.sse, method)
        self.visualizer.visualization_Silhouette(self.silhouetteScores, method)
        self.visualizer.visualization_Rand(self.randScores, method)
        self.visualizer.visualization_NMI(self.normalizedMIScores, method)
        self.visualizer.visualization_AMI(self.adjustedMIScores, method)
        self.visualizer.visualization_VMeasure(self.vMeasureScores, method)
        self.visualizer.visualization_FowlkesMallows(self.FowlkesMallowsScores, method)
    
    def clustering(self, method):
        self.__clean()
        dirPath = os.path.join("./assets/", method)

        interation = self.k_clusters if method != "DBSCAN" else self.eps

        for i in tqdm(interation):
            if method == "Kmeans":
                cluster = KMeans(n_clusters=i, random_state=42)
            elif method == "Agglomerative":
                cluster = AgglomerativeClustering(n_clusters=i)
            elif method == "DBSCAN":
                cluster = DBSCAN(eps=i, min_samples=32)
            elif method == "GMM":
                cluster = GaussianMixture(n_components=i, random_state=42)
            else:
                raise ValueError(f"Invalid method: {method}")
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('clustering', cluster)
            ])
            labels = pipeline.fit_predict(self.trainingSet)
            if method == "Kmeans":
                self.sse.append(pipeline.named_steps['clustering'].inertia_)
            silhouetteScore = silhouette_score(pipeline.named_steps['preprocessor'].transform(self.trainingSet), labels)
            self.silhouetteScores.append(silhouetteScore)
            representations = self.__getMaxClass(labels)
            predictions_representations = [representations[label] for label in labels]
            predictions_binary = self.mlb.transform([[pred] for pred in predictions_representations])
            self.__evaluate(self.y_true.reshape(-1), predictions_binary.reshape(-1))
        self.__visualize(method, dirPath)
    
    def __evaluateSingle(self, y_true, y_pred):
        randScore = rand_score(y_true, y_pred)
        print(f"Rand Index: {randScore}")
        nMIScore = normalized_mutual_info_score(y_true, y_pred)
        print(f"Normalized Mutual Information: {nMIScore}")
        aMIScore = adjusted_mutual_info_score(y_true, y_pred)
        print(f"Adjusted Mutual Information: {aMIScore}")
        vMeasurScore = v_measure_score(y_true, y_pred)
        print(f"V-measure: {vMeasurScore}")
        FMScore= fowlkes_mallows_score(y_true, y_pred)
        print(f"Fowlkes-Mallows Score: {FMScore}")

    def __visualizeSingle(self, method, dirPath, confusionMatrixList):
        self.visualizer = Visualizer(self.k_clusters, self.eps, method, dirPath)
        self.visualizer.visualization_ConfusionMatrix(confusionMatrixList)
        if method == "Hierarchical":
            self.visualizer.visualization_Dendrogram(self.pipeline.named_steps['preprocessor'].transform(self.trainingSet))
    
    def bestCluster(self, method, param):
        dirPath = os.path.join("./assets/", method)
        
        if method == "Kmeans":
            cluster = KMeans(n_clusters=param, random_state=42)
        elif method == "Agglomerative":
            cluster = AgglomerativeClustering(n_clusters=param)
        elif method == "DBSCAN":
            cluster = DBSCAN(eps=param, min_samples=32)
        elif method == "GMM":
            cluster = GaussianMixture(n_components=param, random_state=42)
        else:
            raise ValueError(f"Invalid method: {method}")
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('clustering', cluster)
        ])
        print("##############################")
        print(f"Method: {method}")
        if method == "DBSCAN":
            print(f"optimal eps: {param}")
        else:
            print(f"optimal number of clusters: {param}")
        labels = self.pipeline.fit_predict(self.trainingSet)
        silhouetteScore = silhouette_score(self.pipeline.named_steps['preprocessor'].transform(self.trainingSet), labels)
        representations = self.__getMaxClass(labels)
        predictions_representations = [representations[label] for label in labels]
        predictions_binary = self.mlb.transform([[pred] for pred in predictions_representations])
        confusionMatrixList = multilabel_confusion_matrix(self.y_true, predictions_binary)
        macro_f1 = f1_score(self.y_true, predictions_binary, average='macro')
        print(f"Macro F1: {macro_f1}")

        print(f"Silhouette Score: {silhouetteScore}")
        self.__evaluateSingle(self.y_true.reshape(-1), predictions_binary.reshape(-1))
        self.__visualizeSingle(method, dirPath, confusionMatrixList)

    def featureSelection(self, columns_drop):
        self.trainingSet = self.trainingSet.drop(columns=columns_drop)
        # print(f"Training set: {self.trainingSet}")
        # print(f"Training set shape: {self.trainingSet.shape}")
    
    def setPreprocessor(self, preprocessor):
        self.preprocessor = preprocessor