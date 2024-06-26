import numpy as np
import pandas as pd
import sys
import os
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_estimators = {}
        for model_name, model in models.items():
            print(f"Training {model_name}")

            # Get parameters for the current model
            para = params[model_name]

            # Perform grid search
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Use the best estimator from grid search
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

            # Calculate R2 score
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            best_estimators[model_name] = best_model
        return report, best_estimators
    except Exception as e:
        raise CustomException(e, sys.exc_info()[2])
