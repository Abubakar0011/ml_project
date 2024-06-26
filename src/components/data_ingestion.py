import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.IngestionConfig = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("The data ingestion begin.")
        try:
            df = pd.read_csv('notebook/Data/stud.csv')
            logging.info('The data is reading to dataframe.')

            os.makedirs(os.path.dirname(self.IngestionConfig.train_data_path), exist_ok=True)
            df.to_csv(self.IngestionConfig.raw_data_path, index=False, header=True)

            logging.info('The train test split initiated.')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.IngestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.IngestionConfig.test_data_path, index=False, header=True)
            logging.info('Data Ingestion completed.')

            return (
                self.IngestionConfig.train_data_path,
                self.IngestionConfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    my_obj = DataIngestion()
    my_obj.init_data_ingestion()


        