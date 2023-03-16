# ml_project

python3 -m venv .venv

source .venv/bin/activate



 mkdir src
  - __init__.py

create setup.py



mkdir src/components
  - __init__.py
  - data_ingestion.py  for reading the data
  - data_transformation.py for transforming the raw data to numerics or one-hot
  - model_trainer.py
  - model_pusher.py

mkdir src/pipeline
  - train_pipeline.py
  - predict_pipeline.py
  - __init__py