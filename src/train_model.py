import sys
import os 
from utils import CustomException,logging,save_object,evaluate_models
from process import DataIngestion,DataTransformartion

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
import sklearn.model_selection 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('models','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train,y_train,X_test,y_test,preprocessor_path):
        try:
            models= {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(),
                "KNN": KNeighborsClassifier(),
                "Logistic Regressor": LogisticRegression()
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models= models)
            
            #get the model with the highest score
            best_model_score = max(sorted(model_report.values()))
            
            #get the best model name
            best_model = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        except Exception as e:
            raise CustomException(e,sys)