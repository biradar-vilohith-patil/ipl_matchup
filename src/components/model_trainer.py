import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        candidates = {
            "RandomForest": RandomForestClassifier(
                n_estimators=250,
                max_depth=12,            
                min_samples_leaf=10,      
                class_weight="balanced_subsample", # Crucial for probability calibration
                random_state=42,
                n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}

        for name, model in candidates.items():
            logging.info(f"Training {name} ...")
            model.fit(X_train, y_train)
            preds   = model.predict(X_test)
            acc     = accuracy_score(y_test, preds)
            results[name] = (model, acc)
            logging.info(f"{name} → Test Accuracy: {acc:.4f}")
            print(f"\n{'='*50}")
            print(f"  {name}  |  Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds))

        return results

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test,  y_test  = test_array[:,  :-1], test_array[:,  -1]

            results = self.evaluate_models(X_train, y_train, X_test, y_test)

            best_name, (best_model, best_acc) = max(
                results.items(), key=lambda kv: kv[1][1]
            )

            logging.info(f"Best model: {best_name} with accuracy {best_acc:.4f}")
            print(f"\n✅ Best model selected: {best_name}  (accuracy={best_acc:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved to artifacts/model.pkl")

            return best_acc

        except Exception as e:
            raise CustomException(e, sys)