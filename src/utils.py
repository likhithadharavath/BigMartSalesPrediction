import numpy as np
import pandas as pd
import sys
import os
import dill
from sklearn.metrics import r2_score

from src.exception  import CustomException

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_path:
            dill.dump(obj,file_path)

    except Exception as e:
        raise CustomException(e,sys)
    
'''def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.preddict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        return CustomException(e, sys)'''
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
        return report
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        return {}
