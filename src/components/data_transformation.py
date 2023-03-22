import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # missing value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # to create pkl file
        '''
            Data transformation function 
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]

            categorial_columns = ["gender", 
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"
                                ]
            
            # Create a Pipelimne
            num_pipeline = Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scalar", StandardScaler())         
                                ]
                            )
            
            cat_pipeline = Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("one_hot_encoder", OneHotEncoder()),
                                    ("scalar", StandardScaler(with_mean=False))
                                ]
                            )
            logging.info("Numerical Columns Standard Scaling Completed")

            logging.info("Categorical Columns Encoding Completed")

            preprocessor = ColumnTransformer(
                                        [
                                            ("num_pipeline", num_pipeline, numerical_columns),
                                            ("cat_pipeline", cat_pipeline, categorial_columns)
                                        ]

                            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and test data completed")
            logging.info("Obtaining pre processing object")

            preprocesser_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test data frame ")

            input_feature_train_arr = preprocesser_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesser_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving Preprocessing objects")

            save_object( file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocesser_obj)
            
            logging.info("Saved Preprocessing objects")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)

