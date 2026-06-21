import pandas as pd
from preprocess import preprocess_pipeline, segment_phases
from features import extract_all_features
import warnings
warnings.filterwarnings('ignore')

print("Generating mock data...")
import generate_mock_data
generate_mock_data.os.makedirs('data', exist_ok=True)
df = generate_mock_data.create_mock_ppg(profile="normal")
df.to_csv("data/mock_normal_ppg.csv", index=False)

print("Running pipeline...")
df_processed = preprocess_pipeline("data/mock_normal_ppg.csv", fs=100.0)

segments = segment_phases(df_processed)
for phase_name, phase_df in segments.items():
    print(f"Phase: {phase_name}, points: {len(phase_df)}")
    if len(phase_df) > 500:
        feat = extract_all_features(phase_df['filtered_ppg'].values, fs=100.0)
        print(f"  Features: {feat}")

print("Pipeline successful!")
