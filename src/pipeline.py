from src.utils import load_object, CustomException, logging,dill
import sys 
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        """_summary_
        This method apply sequential steps to predict data and return prediction 
        """
        try:
            #assign model 
            model_path = 'models/model.pkl'
            model=load_object(file_path=model_path)
            
            #assign data transformer
            preprocessor_path='models/preprocessor.pkl'
            preprocessor=load_object(file_path=preprocessor_path)
            transformerd_data= preprocessor.transform(features)
            prediction= model.predict(transformerd_data)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
                 age:float,
                 balance:float,
                 hascrcard:float,
                 isactivemember:float,
                 estimatedsalary:float,
                 geography:object,
                 gender:object,
                 tenure:int,
                 numofproducts:int
                 ):
        self.age =age
        self.balance= balance
        self.hascrcard= hascrcard
        self.isactivemember= isactivemember
        self.estimatedsalary= estimatedsalary
        self.geography= geography
        self.gender= gender
        self.tenure= tenure
        self.numofproducts= numofproducts
        
    def get_data_as_df(self):
        """ 
        This method saves data into dictionary and transform to DataFrame format for prediction
        """
        try:
            custom_input_dict = {   'Age': [self.age],
                                    'Balance': [self.balance],
                                    'HasCrCard': [self.hascrcard],
                                    'IsActiveMember': [self.isactivemember],
                                    'EstimatedSalary': [self.estimatedsalary],
                                    'Geography': [self.geography],
                                    'Gender': [self.gender],
                                    'Tenure': [self.tenure],
                                    'NumOfProducts': [self.numofproducts]
                                        
            }
            return pd.DataFrame(custom_input_dict)
        except Exception as e:
            raise CustomException(e,sys)