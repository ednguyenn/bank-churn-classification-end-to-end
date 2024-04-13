import sys
import os 
from utils import CustomException,logging,save_object,evaluate_models
from dataclasses import dataclass


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
import sklearn.model_selection 
from process import DataIngestion,DataTransformartion

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('models','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
            
    def initiate_model_trainer(self, X_train,y_train,X_test,y_test):
        try:
            #list out models and corresponding params will be used
            models= {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier()
            }
            params={
                "Random Forest":{
                                #'bootstrap': [True],
                                #'max_depth': [80, 90, 100, 110],
                                #'max_features': [2, 3],
                                #'min_samples_leaf': [3, 4, 5],
                                #'min_samples_split': [8, 10, 12],
                                'n_estimators': [100, 200, 300]
                },
                
                "XGBoost":{
                    #'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
                
            }
            #evaluate different model then save them into a dictionary
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models= models,params=params)
            
            #get the model with the highest score
            best_model_score = max(sorted(model_report.values()))
            
            #get the best model name
            best_model_name  = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model= models[best_model_name]
            
            #setting the baseline threshold for the best model 
            if best_model_score<0.7:
                raise CustomException("No best model is not found")
            
            logging.info("Found the best model for the dataset")
            
            #save modelTrainer into destination folder
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return print(f"Best model auc_roc_score is : {best_model_score}")
        except Exception as e:
            raise CustomException(e,sys)