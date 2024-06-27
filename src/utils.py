import os
import sys
import dill
import pandas as pd

from src.exception import customException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise customException(e,sys)

def evaluate_Model(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            if param:
                gs=GridSearchCV(model,param,cv=3)
                gs.fit(X_train,y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_acc=r2_score(y_train,y_train_pred)
            test_model_acc=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_acc
        
        return report

    except Exception as e:
        raise customException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise customException(e,sys)

def get_column_categories():
    try:
        raw_data_path=os.path.join("artifacts","raw_data.csv")
        df=pd.read_csv(raw_data_path)

        # Categories inside different categorical columns
        brand=df['brand'].value_counts().index.tolist()
        model=df['model'].value_counts().index.tolist()
        fuel_type=df['fuel_type'].value_counts().index.tolist()
        engine=df['engine'].value_counts().index.tolist()
        transmission=df['transmission'].value_counts().index.tolist()
        ext_col=df['ext_col'].value_counts().index.tolist()
        int_col=df['int_col'].value_counts().index.tolist()

        return({
            "brand":brand,
            "model":model,
            "fuel_type":fuel_type,
            "engine":engine,
            "transmission":transmission,
            "ext_col":ext_col,
            "int_col":int_col
        })

    except Exception as e:
        raise customException(e,sys)
    
    
