import os 
import sys 

import numpy as np 
import pandas as pd 
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) 
    
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models:dict):
    try:
        report = {}

        for i in range(len(list(models.values()))):
            model = list(models.values())[i]

            train_model = model.fit(X_train,y_train)

            y_pred = train_model.predict(X_test)

            test_score = r2_score(y_test,y_pred)


            report[list(models.keys())[i]] = test_score

        return report 

    except Exception as e:
        raise CustomException(e,sys)    