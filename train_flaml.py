import pandas as pd
from flaml import AutoML
import pickle
import os

def train_and_save_model(data_path="PPG_Feature_Dataset.csv", model_path="flaml_pots_model.pkl"):
    print(f"Loading existing data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Target is 'label'
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label']
    
    print("Class distribution:")
    print(y.value_counts())
    
    automl = AutoML()
    automl_settings = {
        "time_budget": 30, # 30 seconds
        "metric": 'accuracy',
        "task": 'classification',
        "log_file_name": 'flaml.log',
        "seed": 42,
        "verbose": 1 
    }
    
    print("Training FLAML AutoML model on Feature Dataset...")
    automl.fit(X_train=X, y_train=y, **automl_settings)
    
    print(f"Best estimator: {automl.best_estimator}")
    
    # Save the best model
    with open(model_path, 'wb') as f:
        pickle.dump(automl, f)
        
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
