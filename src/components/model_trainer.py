import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing arrays")

            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )

            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )


            # ---------------------------------------------------
            # MODEL INITIALIZATION
            # ---------------------------------------------------

            logging.info("Initializing RandomForestClassifier")

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )


            # ---------------------------------------------------
            # TRAIN MODEL
            # ---------------------------------------------------

            logging.info("Training model")

            model.fit(X_train, y_train)


            # ---------------------------------------------------
            # PREDICTIONS
            # ---------------------------------------------------

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model accuracy: {accuracy}")

            print("Model Accuracy:", accuracy)


            print("\nClassification Report\n")
            print(classification_report(y_test, y_pred))


            # ---------------------------------------------------
            # SAVE MODEL
            # ---------------------------------------------------

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model saved successfully")


            return accuracy


        except Exception as e:
            raise CustomException(e, sys)