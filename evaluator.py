from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Evaluator:
    
    X_test = None
    y_test = None
    mlb = MultiLabelBinarizer()

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        if self.X_test is None or self.y_test is None:
            raise ValueError(f"X_test or y_test is None")
        else:
            self.__labelEncoding()
    
    def __labelEncoding(self):
        """
        label encoding for y_test
        args:
            --None
        return:
            --None
        """
        label_list = self.mlb.fit_transform(self.y_test.values.ravel())
        self.y_test = label_list
    
    def evaluate(self, model, modelName):
        """
        evaluate the model, including accuracy, f1 score, confusion matrix
        args:
            --model: model to be evaluated
        return:
            --None
        """
        print(f"******************************************")
        print(f"Evaluating model {modelName}...")
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc}")

        macro_f1 = f1_score(self.y_test, y_pred, average='macro')
        print(f"Macro F1: {macro_f1}")

        """
        test region for co-occurrence confusion matrix, it seems that the co-occurrence matrix is not suitable.
        """
        """
        # confusion matrix for overall classes(co-occurrence)
        # [i, j] means the number of samples that are labeled as class i and predicted as class j
        n_classes = self.y_test.shape[1]
        confusionMatrix_all = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(n_classes):
            for j in range(n_classes):
                confusionMatrix_all[i, j] = np.sum((self.y_test[:, i] == 1) & (y_pred[:, j] == 1))

        class_labels = [f"Class {i}" for i in range(n_classes)]
        confusion_df = pd.DataFrame(confusionMatrix_all, index=class_labels, columns=class_labels)
        # print("Multi-label Confusion Matrix (Co-occurrence):")
        # print(confusion_df)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title(f"Confusion Matrix for Multi-Label Classification with {modelName}")
        plt.ylabel(f"True Label")
        plt.xlabel(f"Predicted Label")
        plt.savefig(f"assets/{modelName}/confusionMatrix_{modelName}_all.png")
        # plt.show()
        plt.close()
        """

        confusion_matrices = multilabel_confusion_matrix(self.y_test, y_pred)

        for i, cm in enumerate(confusion_matrices):
            dispoly = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
            dispoly.plot(cmap="Blues")
            plt.title(f"Confusion Matrix for Label {i} with {modelName}")
            plt.xlabel('Predicted') 
            plt.ylabel('True')
            plt.savefig(f"assets/{modelName}/confusionMatrix_{modelName}_label{i}.png")
            #plt.show()
            plt.close()
        
    def permutationImportance(self, model, modelName):
        """
        calculate the permutation importance of the model
        args:
            --model: pipeline including preprocessor and classifier
        return:
            --None
        """
        print(f"******************************************")
        print(f"Calculating permutation importance for model {modelName}...")
        model.fit(self.X_test, self.y_test)
        result = permutation_importance(
            model, self.X_test, self.y_test, scoring="f1_weighted", n_repeats=10, random_state=42, n_jobs=2
        )
        importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)
        print("Feature Importances:\n", importance_df)
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{modelName}_Permutation Importance')
        plt.savefig(f'assets/FeatureImportance/{modelName}/PermutationImportance_{modelName}.png')
        plt.show()
        plt.close()