import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads PPG data from a CSV file.
    Expected columns: time, ppg, phase
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def apply_bandpass_filter(data: np.ndarray, lowcut=0.5, highcut=8.0, fs=100.0, order=4) -> np.ndarray:
    """
    Applies a Butterworth band-pass filter to the PPG signal.
    lowcut: 0.5 Hz (removes baseline wander)
    highcut: 8.0 Hz (removes high-frequency noise, most PPG info is below 8Hz)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization of the PPG signal.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return normalized_data

def segment_phases(df: pd.DataFrame) -> dict:
    """
    Segments the dataframe into baseline, transition, and standing phases based on 'phase' column.
    If 'phase' isn't available, we assume the first 100s is baseline, 100-160s transition, >160s standing
    (assuming fs=100Hz as default from mock).
    """
    segments = {}
    
    if 'phase' in df.columns:
        for phase_name in df['phase'].unique():
            segments[phase_name] = df[df['phase'] == phase_name].copy()
    else:
        # Fallback segmentation based on time mapping if 'phase' column is missing
        if 'time' in df.columns:
            baseline = df[df['time'] < 100]
            transition = df[(df['time'] >= 100) & (df['time'] < 160)]
            standing = df[df['time'] >= 160]
            
            segments['baseline'] = baseline
            segments['transition'] = transition
            segments['standing'] = standing
            
    return segments

def preprocess_pipeline(filepath: str, fs: float = 100.0) -> pd.DataFrame:
    """
    Complete preprocessing pipeline: load -> filter -> normalize.
    Returns the dataframe with an additional 'filtered_ppg' column.
    """
    df = load_data(filepath)
    if df is None:
        return None
        
    if 'ppg' not in df.columns:
        raise ValueError("CSV must contain a 'ppg' column.")
        
    raw_ppg = df['ppg'].values
    
    # Filter
    filtered = apply_bandpass_filter(raw_ppg, fs=fs)
    
    # Normalize
    normalized = normalize_signal(filtered)
    
    df['filtered_ppg'] = normalized
    
    return df
