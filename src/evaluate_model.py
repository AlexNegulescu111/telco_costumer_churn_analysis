from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

results = []
confusion = []

def collect_scores(y_true, y_pred, model_name):
    scores = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }
    confusion.append(confusion_matrix(y_true, y_pred))
    results.append(scores)

def compare_models():
    for i, result in enumerate(results):
        print(f"confusion matrix for {result['Model']}")
        print(confusion[i])
    return pd.DataFrame(results)


