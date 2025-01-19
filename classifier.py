from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd
import numpy as np

class Classifier:

    columns_nominal = ['key', 'mode', 'time_signature']
    columns_numeric = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    columns_label = ['genre_list']

    mlb = MultiLabelBinarizer()
    preprocessor = None
    X_train = None
    y_train = None
    pipeline = None
    model_rf = None
    model_SVM = None
    model_linearSVM = None
    model_decisionTree = None
    model_naiveBayes = None
    model_knn = None

    def __init__(self, preprocessor, X_train, y_train):
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.y_train = y_train
        if self.preprocessor is None or self.y_train is None or self.X_train is None:
            raise ValueError(f"preprocessor or y_train or X_train is None")
        else:
            self.__labelEncoding()
    
    def __labelEncoding(self):
        """
        label encoding for y_train
        args:
            --None
        return:
            --None
        """
        label_list = self.mlb.fit_transform(self.y_train.values.ravel())
        self.y_train = label_list
    
    def randomForest(self):
        """
        random forest classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_rf = load('models/random_forest_model.joblib')
            print(f"Model loaded from 'random_forest_model.joblib'")
        except:
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            ovr = OneVsRestClassifier(rf)
            param_grid = {
                'classifier__estimator__n_estimators': [50, 100, 150, 200],
                'classifier__estimator__max_depth': [None, 10, 20],
            }

            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', ovr)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_rf = grid_search.best_estimator_
            self.model_rf.fit(self.X_train, self.y_train)

            dump(self.model_rf, 'models/random_forest_model.joblib')
            print(f"Model saved as 'random_forest_model.joblib'")
        print(f"random forest classifier done")

    def SVM(self):
        """
        support vector machine classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_SVM = load('models/SVM_model.joblib')
            print(f"Model loaded from 'SVM_model.joblib'")
        except:
            svc = SVC(random_state=42, class_weight='balanced')
            ovr = OneVsRestClassifier(svc)
            param_grid = {
                'classifier__estimator__C': [0.01, 0.1, 1, 10, 100],
            }

            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', ovr)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_SVM = grid_search.best_estimator_
            self.model_SVM.fit(self.X_train, self.y_train)

            dump(self.model_SVM, 'models/SVM_model.joblib')
            print(f"Model saved as 'SVM_model.joblib'")
        print(f"support vector machine classifier done")

    def linearSVM(self):
        """
        linear support vector machine classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_linearSVM = load('models/linearSVM_model.joblib')
            print(f"Model loaded from 'linearSVM_model.joblib'")
        except:
            lsvc = LinearSVC(random_state=42, class_weight='balanced', max_iter=100000)
            ovr = OneVsRestClassifier(lsvc)
            param_grid = {
                'classifier__estimator__C': [0.1, 1, 10, 100, 1000, 10000],
            }
            
            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', ovr)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_linearSVM = grid_search.best_estimator_
            self.model_linearSVM.fit(self.X_train, self.y_train)

            dump(self.model_linearSVM, 'models/linearSVM_model.joblib')
            print(f"Model saved as 'linearSVM_model.joblib'")
        print(f"linear support vector machine classifier done")

    def DecisionTree(self):
        """
        decision tree classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_decisionTree = load('models/DecisionTree_model.joblib')
            print(f"Model loaded from 'decision_tree_model.joblib'")
        except:
            dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__criterion': ['gini', 'log_loss'],
                'classifier__splitter': ['best', 'random'],
            }

            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', dt)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_decisionTree = grid_search.best_estimator_
            self.model_decisionTree.fit(self.X_train, self.y_train)

            dump(self.model_decisionTree, 'models/DecisionTree_model.joblib')
            print(f"Model saved as 'decision_tree_model.joblib'")
        print(f"decision tree classifier done")

    def NaiveBayes(self):
        """
        naive bayes classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_naiveBayes = load('models/NaiveBayes_model.joblib')
            print(f"Model loaded from 'naive_bayes_model.joblib'")
        except:
            nb = GaussianNB()
            ovr = OneVsRestClassifier(nb)
            param_grid = {
                'classifier__estimator__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
            }

            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', ovr)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_naiveBayes = grid_search.best_estimator_
            self.model_naiveBayes.fit(self.X_train, self.y_train)

            dump(self.model_naiveBayes, 'models/NaiveBayes_model.joblib')
            print(f"Model saved as 'naive_bayes_model.joblib'")
        print(f"naive bayes classifier done")

    def KNN(self):
        """
        K-nearest neighbors classifier
        args:
            --None
        return:
            --None
        """
        try:
            self.model_knn = load('models/KNN_model.joblib')
            print(f"Model loaded from 'KNN_model.joblib'")
        except:
            knn = KNeighborsClassifier(weights='distance')
            ovr = OneVsRestClassifier(knn)
            param_grid = {
                'classifier__estimator__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17],
            }

            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', ovr)
            ])

            grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='f1_weighted')
            grid_search.fit(self.X_train, self.y_train)

            print("Best hyperparameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            self.model_knn = grid_search.best_estimator_
            self.model_knn.fit(self.X_train, self.y_train)

            dump(self.model_knn, 'models/KNN_model.joblib')
            print(f"Model saved as 'KNN_model.joblib'")
        print(f"K-nearest neighbors classifier done")

    def getPipeline(self, modelName):
        pipeline = None
        if modelName == 'RandomForest':
            pipeline = self.model_rf
            self.model_rf = None
        elif modelName == 'SVM':
            pipeline = self.model_SVM
            self.model_SVM = None
        elif modelName == 'LinearSVM':
            pipeline = self.model_linearSVM
            self.model_linearSVM = None
        elif modelName == 'DecisionTree':
            pipeline = self.model_decisionTree
            self.model_decisionTree = None
        elif modelName == 'NaiveBayes':
            pipeline = self.model_naiveBayes
            self.model_naiveBayes = None
        elif modelName == 'KNN':
            pipeline = self.model_knn
            self.model_knn = None
        else:
            raise ValueError(f"Model name {modelName} is not valid")
        return pipeline
    
    def getFeatureImportance(self, model, modelName):
        """
        get feature importance for the model
        args:
            --model: pipeline including preprocessor and classifier
            --model_name: name of the model
        return:
            --None
        """
        feature_importances = None
        plotTitle = None
        numericFeatures = self.columns_numeric
        categoricalFeatures = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(self.columns_nominal)
        allFeatures = np.concatenate([numericFeatures, categoricalFeatures])
        aggregate_importances = np.zeros(len(allFeatures))

        for i in range(len(model.named_steps['classifier'].estimators_)):
            print(f"************************************************************")
            print(f"Feature importance for estimator {i}")
            if modelName == 'RandomForest':
                feature_importances = model.named_steps['classifier'].estimators_[i].feature_importances_
                aggregate_importances += feature_importances
                plotTitle = 'Feature Importance (MDI) with RandomForest'
            elif modelName == 'LinearSVM':
                feature_importances = np.abs(model.named_steps['classifier'].estimators_[i].coef_[0])
                aggregate_importances += feature_importances
                plotTitle = 'Feature Importance (Weights) with LinearSVM'
            else:
                raise ValueError(f"Model name {modelName} is not valid")
            importance_df = pd.DataFrame({
                'Feature': allFeatures,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            print("Feature Importances:\n", importance_df)
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(plotTitle)
            plt.savefig(f'assets/FeatureImportance/{modelName}/MDI_{modelName}_estimator_{i}.png')
            # plt.show()
            plt.close()
        print(f"************************************************************")
        print(f"Feature importance for estimator {i}")
        aggregate_importances /= len(model.named_steps['classifier'].estimators_)
        importance_df = pd.DataFrame({
            'Feature': allFeatures,
            'Importance': aggregate_importances
        }).sort_values(by='Importance', ascending=False)
        print("Aggregate Feature Importance Across All Classes:\n", importance_df)
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f"Aggregate Feature Importance with {modelName}")
        plt.savefig(f'assets/FeatureImportance/{modelName}/AggregateFeature_{modelName}.png')
        plt.show()
        plt.close()