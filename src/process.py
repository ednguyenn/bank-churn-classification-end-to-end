import os 
import sys
from utils import CustomException, logging,save_object
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data/raw',"train.csv")
    test_data_path: str=os.path.join('data/raw',"test.csv")
    raw_data_path: str=os.path.join('data/raw',"raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df = pd.read_csv('data/raw/kaggle_train.csv')
            logging.info("Read the raw dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df,test_size=0.2, random_state=112)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

@dataclass            
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('models',"preprocessor.pkl")
    
class DataTransformartion:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
            
    def get_data_transformer_object(self):
        """
        This method is responsible for data tranformation
        """
        try:
            numerical_columns = ['Age','Balance','HasCrCard','IsActiveMember','EstimatedSalary']
            categorical_columns = ['Geography','Gender','Tenure','NumOfProducts']
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oh_encoder",OneHotEncoder())
                ]
            )
            
            logging.info("Columns transformed !")
            
            preprocessor=ColumnTransformer(
                [("numerical_pipeline",num_pipeline,numerical_columns),
                 ("categorical_pipeline",categorical_pipeline,categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data")
                                         
            target_column_name="Exited"
            
            X_train= train_df.drop(columns=[target_column_name],axis=1)
            y_train= train_df[target_column_name]
            
            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]

            logging.info("Applying preprocessing object on training set and test set")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            X_train_arr= preprocessing_obj.fit_transform(X_train)
            X_test_arr=preprocessing_obj.transform(X_test)
            
            X_arr = np.c_[X_train_arr,np.array(y_train)]
            y_arr = np.c_[X_test_arr,np.array(y_test)]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                X_arr,
                y_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)      




if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformartion()
    X_arr,y_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
