#read the data here
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read Dataset as dataframes pandas")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split intiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            #print(train_set)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train test split Done and saved")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)


        except Exception as e:
            raise ChildProcessError(e, sys)


