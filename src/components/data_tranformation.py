from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import sys
import os
import numpy as np 
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_object_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "launch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("categorical columns")
            logging.info("numerical columns")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("train and test data read.")

            logging.info("To obtain the preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            target_col = 'math_score'
            numerical_cols = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying the preprocessing obect on train and test dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("processing object saved")
            save_object(
                file_path = self.data_transformation_config.processor_object_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr, test_arr, self.data_transformation_config.processor_object_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)



        
        