from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger import logger


class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }

    def select_best_model(self, trained_models, X_test, y_test, metric="ROC AUC"):
        best_model = None
        best_metric_val = -1.0
        best_name = ""
        evaluation_results = {}

        for name, model in trained_models.items():
            metrics = self.evaluate(model, X_test, y_test)
            evaluation_results[name] = metrics
            logger.info("[%s] %s: %.4f", name, metric, metrics[metric])
            if metrics[metric] > best_metric_val:
                best_metric_val = metrics[metric]
                best_model = model
                best_name = name

        logger.info("Best model: %s (%s: %.4f)", best_name, metric, best_metric_val)
        return best_name, best_model, evaluation_results
