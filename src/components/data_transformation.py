from pandas.core.arrays import categorical
import os
import sys
import numpy as np 
import pandas as pd  
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    

class  DataTransformation:
    '''
    This function is used to transform the data into the required format    
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 

    def get_data_transformation(self):
        try:
            numeric_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

            
    def initiate_data_transformation(self,train_data_path: str,test_data_path: str):
        try:
            logging.info("Data Transformation started")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path) 
            logging.info("Read test and train data")
            logging.info("Obtaining preprocessor")

            preprocessor_obj = self.get_data_transformation()

            target_column = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessor on train and test data")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) 

            logging.info("Applied Preprocessor on train and test data")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Concatenated train and test data")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("saved preprocessor object")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

