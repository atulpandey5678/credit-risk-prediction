from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.logger import logger


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42, class_weight="balanced"
            ),
            "XGBoost": XGBClassifier(
                eval_metric="logloss", random_state=42, verbosity=0
            ),
        }
        self.params = {
            "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0]},
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "scale_pos_weight": [1, 3, 5],
            },
        }

    def train_all(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            logger.info("Training %s...", name)
            grid = GridSearchCV(
                model, self.params[name], cv=3, scoring="roc_auc", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            logger.info("Best params for %s: %s", name, grid.best_params_)
            trained_models[name] = grid.best_estimator_
        return trained_models
