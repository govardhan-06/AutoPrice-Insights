import os
import sys

from src.exception import customException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training_phase(self):
        try:
            logging.info("Initiated Training pipeline...")
            data_ingestion = DataIngestion()
            data_ingestion.dataSplitter()

            data_transformation=DataTransformation()
            train_arr,test_arr,_=data_transformation.intiate_dataTransform()

            model_trainer=ModelTrainer()
            model_trainer.intiate_model_trainer(train_arr,test_arr)
            logging.info("Model Training complete")
        except Exception as e:
            raise customException(e,sys)
    