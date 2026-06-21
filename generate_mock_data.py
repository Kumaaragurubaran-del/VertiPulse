import numpy as np
import pandas as pd
import os

def create_mock_ppg(sampling_rate=100, duration=300, profile="normal"):
    """
    duration in seconds.
    Phase 1: 0-100s Baseline (Supine)
    Phase 2: 100-160s Transition
    Phase 3: 160-300s Standing
    
    Profiles:
      - normal: Baseline 65 bpm -> Standing 75 bpm (increase < 30)
      - pots: Baseline 70 bpm -> Standing 115 bpm (increase > 30)
      - mild_pots: Baseline 65 bpm -> Standing 100 bpm (increase > 30)
      - severe_pots: Baseline 75 bpm -> Standing 140 bpm (increase >> 30)
      - borderline: Baseline 70 bpm -> Standing 98 bpm (increase = 28)
    """
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Base frequencies
    if profile == "pots":
        baseline_hr = 70
        stand_hr = 115
    elif profile == "mild_pots":
        baseline_hr = 65
        stand_hr = 100
    elif profile == "severe_pots":
        baseline_hr = 75
        stand_hr = 140
    elif profile == "borderline":
        baseline_hr = 70
        stand_hr = 98
    else: # normal
        baseline_hr = 65
        stand_hr = 75
        
    # Create dynamic HR curve
    hr_curve = np.zeros_like(t)
    
    # Baseline phase
    baseline_idx = t < 100
    hr_curve[baseline_idx] = baseline_hr
    
    # Transition phase (linear interpolated increase with some wobble)
    transition_idx = (t >= 100) & (t < 160)
    hr_curve[transition_idx] = np.interp(t[transition_idx], [100, 160], [baseline_hr, stand_hr])
    
    # Standing phase
    stand_idx = t >= 160
    hr_curve[stand_idx] = stand_hr
    
    # Add random walk noise to HR curve (HRV sim)
    seed_map = {"normal": 42, "pots": 43, "mild_pots": 44, "severe_pots": 45, "borderline": 46}
    np.random.seed(seed_map.get(profile, 42))
    hrv_noise = np.cumsum(np.random.normal(0, 0.05, len(t)))
    # keep hrv within bounds
    hrv_noise = np.clip(hrv_noise, -5, 5)
    hr_curve += hrv_noise
    
    # Convert HR (bpm) to instantaneous phase
    freq = hr_curve / 60.0  # Hz
    phase = 2 * np.pi * np.cumsum(freq) / sampling_rate
    
    # Generate PPG-like signal (Combination of sines)
    # Fundamental frequency
    ppg = np.sin(phase) + 0.5 * np.sin(2 * phase + np.pi/4)
    
    # Add baseline wander (low freq noise)
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.05 * t) + 0.2 * np.sin(2 * np.pi * 0.02 * t)
    
    # Add high frequency noise (motion artifacts mostly in transition)
    hf_noise = np.random.normal(0, 0.1, len(t))
    # increase noise during transition
    hf_noise[transition_idx] *= 3.0
    
    final_ppg = ppg + baseline_wander + hf_noise
    
    # Format to DataFrame
    df = pd.DataFrame({
        'time': t,
        'ppg': final_ppg,
        'phase': ['baseline' if x < 100 else 'transition' if x < 160 else 'standing' for x in t]
    })
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    profiles = ["normal", "pots", "mild_pots", "severe_pots", "borderline"]
    
    for prof in profiles:
        df = create_mock_ppg(profile=prof)
        filename = f'data/mock_{prof}_ppg.csv'
        df.to_csv(filename, index=False)
        print(f"Created {filename}")
