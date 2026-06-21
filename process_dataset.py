import pandas as pd
import numpy as np
from features import extract_all_features
import os

def process_ppg_dataset(input_file="PPG_Dataset.csv", output_file="data/POTS_extracted_dataset.csv", fs=100.0):
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
        
    print(f"Dataset shape: {df.shape}")
    
    # Check if 'Label' is in columns
    if 'Label' not in df.columns:
        print("Warning: 'Label' column not found. Assuming last column is label.")
        label_col = df.columns[-1]
    else:
        label_col = 'Label'
        
    labels = df[label_col].values
    
    # Process signals
    extracted_data = []
    
    # Use only signal columns (assuming all cols except label are signal)
    signal_cols = [c for c in df.columns if c != label_col]
    signals = df[signal_cols].values
    
    print(f"Extracting features for {len(signals)} segments...")
    for i, sig in enumerate(signals):
        if i % 100 == 0:
            print(f"Processed {i}/{len(signals)} segments...")
            
        try:
            # We assume signal is numeric
            sig_numeric = pd.to_numeric(sig, errors='coerce')
            # Handle NaNs from coercion (if any)
            sig_numeric = np.nan_to_num(sig_numeric)
            
            feat = extract_all_features(sig_numeric, fs=fs)
            
            # Map label: Normal -> 0, else -> 1
            orig_label = str(labels[i]).strip()
            # It's case insensitive check for "normal"
            is_abnormal = 0 if orig_label.lower() == 'normal' else 1
            
            feat['Label'] = is_abnormal
            feat['Original_Label'] = orig_label
            extracted_data.append(feat)
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            
    out_df = pd.DataFrame(extracted_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file} with shape {out_df.shape}")

if __name__ == "__main__":
    process_ppg_dataset()
