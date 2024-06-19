import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
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

from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,clean_train_arr,clean_test_arr):
        logging.info('Ititiating model trainer')
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                clean_train_arr[:,:-1],
                clean_train_arr[:,-1],
                clean_test_arr[:,:-1],
                clean_test_arr[:,-1],
            )
            logging.info('Data successfully splitted')
            
            #specify the models
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifer":AdaBoostRegressor(),
            }
            
            #get the best model
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            


        except Exception as e :
            raise CustomException(e,sys)

