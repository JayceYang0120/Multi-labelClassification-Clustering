from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

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