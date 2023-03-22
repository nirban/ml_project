import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    traind_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Splitting Data array into inpyt Variables")
            x_train, y_train, x_test, y_test = ( 
                                            train_array[:, :-1], 
                                            train_array[:,-1],
                                            test_array[:, :-1], 
                                            test_array[:,-1],
                                        )
            
            models = {
                    "Random_Forest": RandomForestRegressor(),
                    "Decision_Tree": DecisionTreeRegressor(),
                    "Gradient_Boosting": GradientBoostingRegressor(),
                    "Linear_Regression": LinearRegression(),
                    "KNN_Regression": KNeighborsRegressor(),
                    "XGBoost_Regression": XGBRegressor(),
                    "CatBoost_Regression": CatBoostRegressor(),
                    "AdaBoost_Regression": AdaBoostRegressor()
                }
            
            model_report:dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            ## get the best model Score from Dict model_report
            best_model_score = max(sorted(model_report.values()))

            ## get the best model name

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if(best_model_score <0.6):
                raise CustomException("Models is below Threshold, Hence Not Found")
            
            logging.info(f"Best Model - {best_model_name} with score {best_model_score} found on both Training and Test data set")

            ## optional create a preprocessor obj

            logging.info(f"Saving the Model File")

            save_object(file_path=self.model_trainer_config.traind_model_file_path, obj=best_model)

            logging.info(f"Saved the Model File")

            predicted = best_model.predict(x_test)

            r2_scores = r2_score(y_test, predicted)

            return r2_scores


        except Exception as e:
            raise CustomException(e, sys)