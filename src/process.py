import os 
import sys
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

from utils import CustomException, logging,save_object,evaluate_models

#a convenient way to define classes to store data
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data/raw',"train.csv")
    test_data_path: str=os.path.join('data/raw',"test.csv")
    raw_data_path: str=os.path.join('data/raw',"raw.csv")
    
#class with data ingestion operation methods
class DataIngestion:
    #allow DataIngestion to access the paths defined in DataIngestionConfig
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """_summary_
        This method will retrieve data from origin source and save it to project directory
        Then perform train_test_split and save them into corresponding folder

        Returns: paths to train and test data
       
        """
        logging.info("Entered data ingestion method or component")
        try:
            #read files from original source (raw data source)
            df = pd.read_csv('data/raw/kaggle_train.csv')
            logging.info("Read the raw dataset as dataframe")
            
            #ensures that the destination directory for storing processed data exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            #save raw data which was read from orginal source to local destination
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            
            #train test split
            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df,test_size=0.2, random_state=112)
            
            #save train and test data into local destination
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
    X_train_data_path: str=os.path.join('data/processed',"X_train.csv")
    y_train_data_path: str=os.path.join('data/processed',"y_train.csv")
    X_test_data_path: str=os.path.join('data/processed',"X_test.csv")
    y_test_data_path: str=os.path.join('data/processed',"y_test.csv")
    
class DataTransformartion:
    #initialize an attribute to save the processed data paths
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
            
    def get_data_transformer_object(self):
        """
        This method generates data proprocessor 
        """
        try:
            #sorted numerical and categorical columnsd
            numerical_columns = ['Age','Balance','HasCrCard','IsActiveMember','EstimatedSalary']
            categorical_columns = ['Geography','Gender','Tenure','NumOfProducts']
            
            #create seperate Pipeline for numerical and categorical columns
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
            
            #create a mutual preprocessor for all columns 
            preprocessor=ColumnTransformer(
                [("numerical_pipeline",num_pipeline,numerical_columns),
                 ("categorical_pipeline",categorical_pipeline,categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        This method will user preproccesor to transform data for model training
        save X_train, y_train, X_test, y_test into data/processed folders 
        """
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
            
            X_train= preprocessing_obj.fit_transform(X_train)
            X_test=preprocessing_obj.transform(X_test)
            
            #save the processed datasets into data/processed folder
            X_train_df = pd.DataFrame(X_train)
            X_train_df.to_csv(self.data_transformation_config.X_train_data_path,index=False,header=True)
            y_train.to_csv(self.data_transformation_config.y_train_data_path,index=False,header=True)
            
            X_test_df = pd.DataFrame(X_test)
            X_test_df.to_csv(self.data_transformation_config.X_test_data_path,index=False,header=True)
            y_test.to_csv(self.data_transformation_config.y_test_data_path,index=False,header=True)
               
               
               
            logging.info("Saved preprocessing object")
            
            #save object into destination folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                X_train,
                y_train,
                X_test,
                y_test
            )
        except Exception as e:
            raise CustomException(e,sys)      




if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformartion()
    X_train,y_train,X_test,y_test=data_transformation.initiate_data_transformation(train_data,test_data)

    
    #import ModelTrainer here to avoid circular import errror between process.py and train_model.py
    from train_model import ModelTrainer
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train,y_train,X_test,y_test))