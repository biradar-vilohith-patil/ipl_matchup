import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass


    def run_pipeline(self):

        try:

            logging.info("Training pipeline started")




            ingestion = DataIngestion()

            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")



            transformation = DataTransformation()

            train_arr, test_arr, preprocessor_path = transformation.data_trans(
                train_data_path,
                test_data_path
            )

            logging.info("Data transformation completed")


         

            trainer = ModelTrainer()

            accuracy = trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info(f"Model training completed with accuracy: {accuracy}")


        except Exception as e:

            raise CustomException(e, sys)



if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()