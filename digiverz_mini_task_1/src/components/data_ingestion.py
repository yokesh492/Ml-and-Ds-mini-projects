import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

#########for debugging purpose
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join('artifacts','data.csv')
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')

 
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Initializing data ingestion')

        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('reading the datasets')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)#artifacts
            #raw data ingest
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('train_test_split initateing')
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)

            #for test
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path=obj.initate_data_ingestion()

    ############## testing the data transformation modelue
    data_transformation = DataTransformation()
    train_arr , test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

    ################ testing the model trainer .py
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))