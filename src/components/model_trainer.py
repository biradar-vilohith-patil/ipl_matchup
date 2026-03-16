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


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# -------------------------------------------------------
# MODEL TRAINER
# -------------------------------------------------------

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # ---------------------------------------------------
    # EVALUATE MULTIPLE MODELS AND PICK THE BEST
    # ---------------------------------------------------

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """
        Trains several candidates and returns a dict of
        {model_name: test_accuracy}.
        """
        candidates = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,            # reduced from 20 → less overfitting
                min_samples_leaf=5,      # prevents tiny leaf splits
                class_weight="balanced", # fixes class imbalance (dots dominate)
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

    # ---------------------------------------------------
    # MAIN TRAINING ENTRY POINT
    # ---------------------------------------------------

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test,  y_test  = test_array[:,  :-1], test_array[:,  -1]

            logging.info(
                f"X_train: {X_train.shape} | X_test: {X_test.shape} | "
                f"Classes: {np.unique(y_train)}"
            )

            # ---------------------------------------------------
            # TRAIN & EVALUATE ALL CANDIDATES
            # ---------------------------------------------------
            results = self.evaluate_models(X_train, y_train, X_test, y_test)

            # ---------------------------------------------------
            # PICK BEST MODEL
            # ---------------------------------------------------
            best_name, (best_model, best_acc) = max(
                results.items(), key=lambda kv: kv[1][1]
            )

            logging.info(f"Best model: {best_name} with accuracy {best_acc:.4f}")
            print(f"\n✅ Best model selected: {best_name}  (accuracy={best_acc:.4f})")

            if best_acc < 0.50:
                logging.warning(
                    "Best model accuracy is below 50%. "
                    "Consider reviewing feature engineering."
                )

            # ---------------------------------------------------
            # CROSS-VALIDATION ON BEST MODEL (sanity check)
            # ---------------------------------------------------
            logging.info("Running 5-fold cross-validation on best model ...")
            cv_scores = cross_val_score(
                best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
            )
            logging.info(
                f"CV scores: {cv_scores}  |  Mean: {cv_scores.mean():.4f} "
                f"± {cv_scores.std():.4f}"
            )
            print(
                f"\n📊 5-Fold CV  →  Mean: {cv_scores.mean():.4f} "
                f"± {cv_scores.std():.4f}"
            )

            # ---------------------------------------------------
            # SAVE BEST MODEL
            # ---------------------------------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved to artifacts/model.pkl")

            return best_acc

        except Exception as e:
            raise CustomException(e, sys)