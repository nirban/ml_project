
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

def main():
    obj = DataIngestion()
    train_data, test_data = obj.intiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2Score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(r2Score)

if __name__ =="__main__":
    main()