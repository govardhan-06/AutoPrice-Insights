import os
import sys
import pandas as pd

from src.exception import customException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_dir = "artifacts/model.pkl"
            model = load_object(file_path=model_dir)

            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Applying the user input data onto the model")

            features_scaled=preprocessor.transform(features)

            logging.info("User Input Data Scaled successfully")

            pred=model.predict(features_scaled)
            
            logging.info("Model prediction generated")

            return pred
        except Exception as e :
            raise customException(e,sys)

class UserInputData:
    def __init__(self,
                 brand:str,
                 model:str,
                 model_year:str,
                 mileage:str,
                 fuel_type:str,
                 engine:str,
                 transmission:str,
                 ext_col:str,
                 int_col:str,
                 accident:str,
                 clean_title:str,
                 ):
        self.brand=brand
        self.model=model
        self.model_year=model_year
        self.mileage=mileage
        self.fuel_type=fuel_type
        self.engine=engine
        self.transmission=transmission
        self.ext_col=ext_col
        self.int_col=int_col
        self.accident=accident
        self.clean_title=clean_title

    def createDataFrame(self):
        try:
            logging.info("Preparing user input data as dataframe")
            user_input={
                "brand":[self.brand],
                "model":[self.model],
                "model_year":[self.model_year],
                "milage":[self.mileage],
                "fuel_type":[self.fuel_type],
                "engine":[self.engine],
                "transmission":[self.transmission],
                "ext_col":[self.ext_col],
                "int_col":[self.int_col],
                "accident":[self.accident],
                "clean_title":[self.clean_title]
            }

            logging.info("Created user input dataframe")
            return(pd.DataFrame(user_input))
        
        except Exception as e:
            raise customException(e,sys)
        