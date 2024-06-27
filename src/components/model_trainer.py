import os
import sys
from dataclasses import dataclass

from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_Model

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def intiate_model_trainer(self,train_arr,test_arr):
        logging.info("Intiate train-test split")
        try:
            X_train,y_train,X_test,y_test=(
            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:,:-1],
            test_arr[:,-1],)

            models={
                    "LinearRegression":LinearRegression(),
                    "DecisionTreeRegressor":DecisionTreeRegressor(),
                    "KNeighborsRegressor":KNeighborsRegressor(),
                    "AdaBoostRegressor":AdaBoostRegressor(),
                    "GradientBoostingRegressor":GradientBoostingRegressor(),
                    "RandomForestRegressor":RandomForestRegressor(),
                    "XGBRegressor":XGBRegressor(),
                }
            
            params = {
                            "DecisionTreeRegressor": {
                                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                            },
                            "RandomForestRegressor": {
                                'n_estimators': [8, 16, 32, 64, 128, 256]
                            },
                            "KNeighborsRegressor":{},
                            "GradientBoostingRegressor": {
                                'learning_rate': [.1, .01, .05, .001],
                                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                'n_estimators': [8, 16, 32, 64, 128, 256]
                            },
                            "LinearRegression": {},
                            "XGBRegressor": {
                                'learning_rate': [.1, .01, .05, .001],
                                'n_estimators': [8, 16, 32, 64, 128, 256]
                            },
                            "AdaBoostRegressor": {
                                'learning_rate': [.1, .01, 0.5, .001],
                                'n_estimators': [8, 16, 32, 64, 128, 256]
                            }
                        }
            
            model_results=evaluate_Model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score=max(sorted(model_results.values()))
            best_model_name=[model for model, score in model_results.items() if score==best_model_score]

        
            logging.info(f"Model training successfull; (Model in use: {best_model_name})")
            best_model=models[best_model_name]

            save_object(
                self.config.model_path,
                best_model
            )

            y_pred=best_model.predict(X_test)
            return {best_model_name:r2_score(y_test,y_pred)}

        except Exception as e:
            raise customException(e,sys)
        