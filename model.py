import pickle
import os
import numpy as np
import pandas as pd
# FLAML model is handled in train_flaml.py

def load_flaml_model(model_path="flaml_pots_model.pkl"):
    """
    Loads the trained FLAML model. You must run train_flaml.py first.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please run `python train_flaml.py` to generate it from the extracted dataset.")

def predict_pots(model, feat_dict):
    """
    Predicts POTS probability using the trained ML model based on standing phase features.
    """
    if not feat_dict:
        return False, 0.0
        
    cols = ['mean', 'std', 'max', 'min', 'range', 'median', 'var', 'num_peaks', 'hr', 'hrv']
    f = [feat_dict.get(c, 0) for c in cols]
    
    # Needs DataFrame for FLAML
    X_test = pd.DataFrame([f], columns=cols)
    
    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]
    
    return bool(prediction == 1), probabilities[1]
