import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str  = os.path.join("artifacts", "test.csv")
    raw_data_path: str   = os.path.join("artifacts", "data.csv")

    # Single source-of-truth for the raw CSV filename
    raw_csv_path: str = os.path.join("data", "processed", "clean_ipl_data.csv")




class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered DataIngestion.initiate_data_ingestion()")
        try:
            df = pd.read_csv(self.ingestion_config.raw_csv_path)
            logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved to artifacts/")

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=None
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,  index=False, header=True)

            logging.info(
                f"Train/test split done → train={len(train_set)}, test={len(test_set)}"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()   # ← fixed method name
    print("Train:", train_data)
    print("Test :", test_data)