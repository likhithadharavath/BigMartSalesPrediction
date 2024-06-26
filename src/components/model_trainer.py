import sys
import traceback

def error_message_detail(error_message, error_detail):
    exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
    return error_message

class CustomException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        self.error_message = error_message_detail(message, error_detail)

    def __str__(self):
        return self.error_message


import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
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

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, clean_train_arr, clean_test_arr):
        logging.info('Initiating model trainer')
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                clean_train_arr[:, :-1],
                clean_train_arr[:, -1],
                clean_test_arr[:, :-1],
                clean_test_arr[:, -1],
            )
            logging.info('Data successfully split')

            # Specify the models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regression": {},  # Linear Regression has no hyperparameters to tune
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "XGB Regressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
            }

            # Get the best model and their corresponding scores
            model_report, best_estimators = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if not model_report:
                raise CustomException("No models were evaluated properly. The model report is empty.", sys.exc_info()[2])

            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = best_estimators[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance.", sys.exc_info()[2])

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score on test data: {r2_square}")

            return r2_square

        except Exception as e:
            _, _, exc_traceback = sys.exc_info()
            raise CustomException(str(e), exc_traceback)
