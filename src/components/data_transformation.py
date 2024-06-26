import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


#create a Data TransformationConfig Class
@dataclass
class DataTransformationConfig():
    preprocessor_file_path=os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
    clean_train_file_path=os.path.join(os.getcwd(),"artifacts","clean_train.csv")
    clean_test_file_path=os.path.join(os.getcwd(),"artifacts","clean_test.csv")
    

#create Data Transformation class
class DataTransformation():
    
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    
    def getPreprocessorObject(self):
        
        # separate columns based on data types
        categorical_cols=['Item_Fat_Content', 'Item_Type','Outlet_Location_Type','Outlet_Type']
        numerical_cols=['Item_Outlet_Sales']


        Fat_categories = ['Low Fat', 'Regular','low fat','LF','reg']
        Item_Type_Categories=['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables','Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
       'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
       'Breads', 'Starchy Foods', 'Others', 'Seafood']
        Location_categories = ['Tier 1', 'Tier 2', 'Tier 3']
        Outlet_Type_categories = ['Supermarket Type1', 'Supermarket Type2','Supermarket Type3','Grocery Store']
        
        ## Numerical Pipeline
        num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='mean')),
            ('scaler',StandardScaler())

            ]

        )

        # Categorical Pipeline
        cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder',OrdinalEncoder(categories=[Fat_categories,Item_Type_Categories,Location_categories,Outlet_Type_categories]))])

        preprocessor=ColumnTransformer([
        ('num_pipeline',num_pipeline,numerical_cols),
        ('cat_pipeline',cat_pipeline,categorical_cols)
        ])
        logging.info('Preprocessor created successfully!')

        return preprocessor
    
    def initiateDataTransformation(self,train_path,test_path):
        logging.info('Data Transformation has started')
        try:
                        
            #read test and train data
            train_df=pd.read_csv(train_path)
            logging.info('Train data read successfully')
            
            test_df=pd.read_csv(test_path)
            logging.info('Test data read successfully')
            print("train:",train_df)
            print("test:",test_df)
                        
            #split dependent and independent features
            if 'Item_Identifier' not in train_df.columns or 'Item_Fat_Content' not in train_df.columns or 'Item_Type' not in train_df.columns or 'Outlet_Identifier' not in train_df.columns or 'Outlet_Location_Type' not in train_df.columns or 'Outlet_Type' not in train_df.columns or 'Item_Outlet_Sales' not in train_df.columns:
                raise KeyError("Columns must be present in the data")

            X_train=train_df[['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']]
            X_test=test_df[['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']]
            y_train=train_df['Item_Outlet_Sales']
            y_test=test_df['Item_Outlet_Sales']
            logging.info('Splitting of Dependent and Independent features is successful')
           
            
            # get preprocessor and pre-process the content  
            preprocessor=self.getPreprocessorObject()
            X_train_arr=preprocessor.fit_transform(X_train)
            logging.info('X_train successfully pre-processed')
            
            X_test_arr=preprocessor.transform(X_test)
            logging.info('X_test successfully pre-processed')
            
            #combine X_train_arr with y_train and vice versa
            clean_train_arr=np.c_[X_train_arr,np.array(y_train)]
            clean_test_arr=np.c_[X_test_arr,np.array(y_test)]
            logging.info('Concatenation of  cleaned arrays is successful')
            
            #save the pre-processor 
            save_obj(self.transformation_config.preprocessor_file_path,preprocessor)
            logging.info('Pre-processor successfully saved')
            
            return(
                clean_train_arr,clean_test_arr
            )        
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":    
    data_transformation=DataTransformation()
    data_transformation.initiateDataTransformation(train_path='artifacts\\train.csv',test_path='artifacts\\test.csv')
  