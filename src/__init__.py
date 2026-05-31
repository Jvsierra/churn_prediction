from src.data_collection import get_input_data
from src.data_preprocessing import cast_yes_no_variables_to_binary
from src.feature_engineering import encode_features
from src.model import train_xgboost, predict_churn
from src.pipeline import run_pipeline