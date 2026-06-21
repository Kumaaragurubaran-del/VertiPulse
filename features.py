import numpy as np
from scipy.signal import find_peaks, welch

def get_peaks(ppg_signal, fs=100.0):
    """
    Detects systolic peaks in the given PPG signal.
    Returns the indices of the peaks.
    """
    # minimum distance between peaks assuming maximum HR of ~200 bpm (3.33 Hz)
    distance = int(fs / 3.5)
    peaks, _ = find_peaks(ppg_signal, distance=distance)
    return peaks

def get_nn_intervals(peaks, fs=100.0):
    """
    Calculates NN intervals (in milliseconds) from peak indices.
    """
    nn_intervals_sec = np.diff(peaks) / fs
    nn_intervals_ms = nn_intervals_sec * 1000.0
    return nn_intervals_ms

def calculate_time_domain(nn_intervals):
    """
    Computes time-domain HRV features: HR, RMSSD, SDNN.
    """
    if len(nn_intervals) < 2:
        return {'hr': 0, 'rmssd': 0, 'sdnn': 0}
        
    nn_diff = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(nn_diff**2))
    sdnn = np.std(nn_intervals)
    hr = 60000.0 / np.mean(nn_intervals) if np.mean(nn_intervals) > 0 else 0
    
    return {'hr': hr, 'rmssd': rmssd, 'sdnn': sdnn}

def calculate_freq_domain(nn_intervals):
    """
    Computes frequency-domain HRV features: LF, HF, LF/HF ratio.
    Uses Welch's method after interpolating NN intervals.
    """
    if len(nn_intervals) < 5:
        return {'lf': 0, 'hf': 0, 'lf_hf_ratio': 0}
        
    # Create absolute time axis in seconds
    time_ax = np.cumsum(nn_intervals) / 1000.0
    
    # Interpolate to uniform sampling rate (e.g., 4 Hz)
    fs_interp = 4.0
    time_uniform = np.arange(time_ax[0], time_ax[-1], 1.0/fs_interp)
    if len(time_uniform) < 5:
        return {'lf': 0, 'hf': 0, 'lf_hf_ratio': 0}
        
    nn_uniform = np.interp(time_uniform, time_ax, nn_intervals)
    
    # Welch's method
    nperseg = min(256, len(nn_uniform))
    freqs, psd = welch(nn_uniform, fs=fs_interp, nperseg=nperseg)
    
    # Frequency bands
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    
    lf = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
    hf = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
    
    lf_hf_ratio = lf / hf if hf > 0 else 0
    
    return {'lf': lf, 'hf': hf, 'lf_hf_ratio': lf_hf_ratio}

def approximate_entropy(U, m=2, r=None):
    """
    Computes approximate entropy of the signal.
    """
    U = np.array(U)
    N = len(U)
    if N <= m + 1:
        return 0
        
    if r is None:
        r = 0.2 * np.std(U)
        
    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = []
        for x_i in x:
            dist = np.max(np.abs(np.array(x) - np.array(x_i)), axis=1)
            C.append(np.sum(dist <= r) / (N - m + 1.0))
        return np.sum(np.log(C)) / (N - m + 1.0)
        
    try:
        return np.abs(_phi(m) - _phi(m + 1))
    except (ValueError, ZeroDivisionError):
        return 0

def extract_all_features(ppg_signal, fs=100.0):
    """
    High-level function to extract all features from a single PPG signal segment.
    """
    peaks = get_peaks(ppg_signal, fs=fs)
    nn_intervals = get_nn_intervals(peaks, fs=fs)
    
    td_feat = calculate_time_domain(nn_intervals)
    fd_feat = calculate_freq_domain(nn_intervals)
    entropy = approximate_entropy(nn_intervals)
    
    features = {
        'hr': td_feat['hr'],
        'rmssd': td_feat['rmssd'],
        'sdnn': td_feat['sdnn'],
        'lf': fd_feat['lf'],
        'hf': fd_feat['hf'],
        'lf_hf_ratio': fd_feat['lf_hf_ratio'],
        'entropy': entropy
    }
    
    return features
