import os
import sys
from dataclasses import dataclass
import datetime
import numpy as np
import pandas as pd
from scipy import stats as stat

from src.exception import customException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin #for custom column transformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

#custom transformer for Age column
class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        now = datetime.datetime.now()
        current_year = now.year
        X['Age'] = current_year - (X['model_year'].astype(int))
        X['Age'] = np.where(X['Age']==0,1,X['Age']) # Replace 0 by 1 to prevent log(0) error
        X['Age'] = np.log(X['Age'])
        return X.drop(columns=['model_year'])

#custom transformer for Price column
class PriceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:,'price'] = [int(i.strip("$").replace(",","")) for i in X['price']]
        return X

#custom transformer for Mileage column
class MileageTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['milage'] = X['milage'].str.replace("mi.", "").str.replace(",", "")
        X['milage'] = pd.to_numeric(X['milage'], errors='coerce').fillna(0).astype(int)
        X['milage'] = np.where(X['milage'] == 0, 1, X['milage'])  # Replace 0 with 1 to avoid log(0)
        if len(X['milage'])==1: ##For user input data only
            transform=[X['milage'][0],1]
            transform,_ = stat.boxcox(transform)
            X['milage']=transform[0]
        else:
            X['milage'],_ = stat.boxcox(X['milage'])
        return X

#custom transformer for fuel_type column
class FuelTypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['fuel_type'] = np.where(X["fuel_type"]=="–","not supported",X["fuel_type"])

        # Number of null values in the 'fuel_type' column
        num_nulls = X['fuel_type'].isnull().sum()

        # Sample non-null 'fuel_type' values
        sampled_values = X['fuel_type'].dropna().sample(num_nulls, random_state=0, replace=True).values

        # Assign the sampled values to the positions where 'fuel_type' is null
        X.loc[X['fuel_type'].isnull(), 'fuel_type'] = sampled_values

        return X

#custom transformer for removing infinity values
class CommonTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[np.isinf(X)] = np.nan  # Replace infinity with NaN
        imp = SimpleImputer(strategy='mean')  # Example imputation strategy
        X_inpute = imp.fit_transform(X)
        X=pd.DataFrame(X_inpute,columns=X.columns)
        return X
    
#Custom transformer for Label Encoding
#LabelEncoder() cannot be used directly in the pipeline
class ColorEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Assuming X is a DataFrame with categorical columns
        self.categories_ = {col: X[col].astype('category').cat.categories.tolist() for col in X.columns}
        return self

    def transform(self, X):
        X_encoded = X.copy()
        
        for col in X.columns:
            X_encoded[col] = X[col].astype('category').cat.codes
            X_encoded[col] = X_encoded[col].where(X[col].isin(self.categories_[col]), -1)
        
        return X_encoded

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_processor(self):
        '''
        This function handles the data transformation phase
        '''
        try:
            # Define the pipeline (for independent features) steps
            cat_column_onehot_title=["clean_title"]
            cat_column_onehot_accident=["accident"]
            num_column_milage=["milage"]
            num_column_year=['model_year']
            cat_column_targetEncode=["brand","model","transmission","engine"]
            cat_column_targetEncode_fueltype=['fuel_type']
            cat_column_labelEncode=["int_col","ext_col"]

            num_column_milage_pipeline=Pipeline(
                steps=[
                    ('mileageTransformer',MileageTransformer()),
                    ('common transform',CommonTransformer()),
                    ('scaler',MinMaxScaler())
                ]
            )

            num_column_year_pipeline=Pipeline(
                steps=[
                    ('ageTranform',AgeTransformer()),
                    ('common transform',CommonTransformer()),
                    ('scaler',MinMaxScaler())
                ]
            )

            cat_column_onehot_accident_pipeline=Pipeline(
                steps=[
                    ('fill Null values', SimpleImputer(strategy='constant',fill_value='None reported')),
                    ('onehot',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first')),
                    ('scaler',MinMaxScaler())
                ]
            )

            cat_column_onehot_cleanTitle_pipeline=Pipeline(
                steps=[
                    ('fill Null values', SimpleImputer(strategy='constant',fill_value='No')),
                    ('onehot',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first')),
                    ('scaler',MinMaxScaler())
                ]
            )

            cat_column_labelEncode_pipeline=Pipeline(
                steps=[
                    ('label encoding',ColorEncoderTransformer()),
                    ('scaler',MinMaxScaler())
                ]
            )

            cat_column_targetEncode_pipeline=Pipeline(
                steps=[
                    ('target encoding',TargetEncoder(smoothing=0.2)),
                    ('scaler',MinMaxScaler())
                ]
            )

            cat_column_targetEncode_fuelType_pipeline=Pipeline(
                steps=[
                    ('fueltype Transformer',FuelTypeTransformer()),
                    ('target encoding',TargetEncoder(smoothing=0.2)),
                    ('scaler',MinMaxScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_col_year pipeline',num_column_year_pipeline,num_column_year),
                    ('num_col_milage pipeline',num_column_milage_pipeline,num_column_milage),
                    ('cat_column_onehot_accident pipeline',cat_column_onehot_accident_pipeline,cat_column_onehot_accident),
                    ('cat_column_onehot_cleanTitle pipeline',cat_column_onehot_cleanTitle_pipeline,cat_column_onehot_title),
                    ('cat_column_labelEncode pipeline',cat_column_labelEncode_pipeline,cat_column_labelEncode),
                    ('cat_column_targetEncode pipeline',cat_column_targetEncode_pipeline,cat_column_targetEncode),
                    ('cat_column_targetEncode_fuelType pipeline',cat_column_targetEncode_fuelType_pipeline,cat_column_targetEncode_fueltype)
                ]
            )

            return(preprocessor)
        
        except Exception as e:
            raise customException(e, sys)
    
    def intiate_dataTransform(self):
        try:
            train_df=pd.read_csv("artifacts/train_data.csv")
            test_df=pd.read_csv("artifacts/test_data.csv")
            logging.info("Read the train and test data")

            logging.info("Loading Preprocessor Object")
            processor=self.get_processor()
            targetTransformer=PriceTransformer()
            logging.info("Processor object loaded")

            target_col='price'

            input_feature_train_df=train_df.drop([target_col],axis=1)
            target_feature_train_df=train_df[[target_col]]

            input_feature_test_df=test_df.drop([target_col],axis=1)
            target_feature_test_df=test_df[[target_col]]

            logging.info("Splitted indepedent and dependent features for both train and test dataset")
            
            logging.info("Applying the fit_transform on the train and test data")
            target_feature_train_arr=targetTransformer.fit_transform(target_feature_train_df)
            target_feature_test_arr=targetTransformer.transform(target_feature_test_df)
            
            logging.info("Successfully transformed dependent features")

            input_feature_train_arr=processor.fit_transform(input_feature_train_df,target_feature_train_df)
            logging.info("Train data transformed")
            input_feature_test_arr=processor.transform(input_feature_test_df)
            logging.info("Transformation successfull")

            train_arr=np.c_[
                input_feature_train_arr,target_feature_train_arr
            ]
            test_arr=np.c_[
                input_feature_test_arr,target_feature_test_arr
            ]

            logging.info("Saving pre-processing object")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=processor
            )
            logging.info("Data Transformation Completed")

            return(
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise customException(e, sys)

if __name__ == "__main__":
    data_transformation_config = DataTransformationConfig()
    
