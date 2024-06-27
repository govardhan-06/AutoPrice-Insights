import os
import pandas as pd
import sys

#Add file path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import customException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train_data.csv')
    test_data_path: str=os.path.join("artifacts","test_data.csv")
    raw_data_path: str=os.path.join("artifacts","raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def dataSplitter(self):
        logging.info("Entering the data ingestion phase...")
        try:
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)

            logging.info("Reading the raw dataset")
            df = pd.read_csv("notebook/data/used_cars.csv")

            df.to_csv(self.config.raw_data_path,index=False,header=True)
            logging.info("Created the raw data file artifacts folder")

            logging.info("Intiated train-test split")
            train_dataset,test_dataset=train_test_split(df,test_size=0.3,train_size=0.7,random_state=42)
            train_dataset.to_csv(self.config.train_data_path,index=False,header=True)
            test_dataset.to_csv(self.config.test_data_path,index=False,header=True)
            logging.info("Train and Test data has been created")
            logging.info("Data Ingestion phase completed successfully")
            
            return(
                train_dataset,
                test_dataset
            )
        
        except Exception as e:
            raise customException(e,sys)