import os 
import sys 
from dataclasses import dataclass 

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models = {
                "Random Forest":RandomForestRegressor(n_jobs=-1),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Classifier":KNeighborsRegressor(n_jobs=-1),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor()}
            
            model_report:dict = evaluate_models(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test,
                                               models=models)
            
            # get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            # get best model name from dict 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)
            
            logging.info("Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted) 
            print(r2_square)
            
        except Exception as e:
            raise CustomException(e,sys)