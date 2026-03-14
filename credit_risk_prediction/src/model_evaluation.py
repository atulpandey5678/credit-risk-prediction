import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    def __init__(self):
        pass
        
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }
        return metrics
        
    def select_best_model(self, trained_models, X_test, y_test, metric_to_optimize="ROC AUC"):
        best_model = None
        best_metric_val = -1
        best_name = ""
        evaluation_results = {}
        
        for name, model in trained_models.items():
            metrics = self.evaluate(model, X_test, y_test)
            evaluation_results[name] = metrics
            print(f"[{name}] {metric_to_optimize}: {metrics[metric_to_optimize]:.4f}")
            
            if metrics[metric_to_optimize] > best_metric_val:
                best_metric_val = metrics[metric_to_optimize]
                best_model = model
                best_name = name
                
        print(f"\nBest Model selected: {best_name} with {metric_to_optimize}: {best_metric_val:.4f}")
        return best_name, best_model, evaluation_results
