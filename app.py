import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from scipy.signal import find_peaks

from preprocess import preprocess_pipeline
from model import load_flaml_model, predict_pots
import visuals
from report_generator import generate_pots_report

st.set_page_config(page_title="POTS Screening & Analytics Dashboard", layout="wide", page_icon="🫀")

@st.cache_resource
def get_model():
    try:
        return load_flaml_model("flaml_pots_model.pkl")
    except Exception:
        return None

model = get_model()

@st.cache_data
def get_data():
    try:
        raw = pd.read_csv("PPG_Feature_Dataset.csv")
        return raw
    except:
        return None

data_df = get_data()

st.title("🫀 POTS Diagnostic Analytics & Screening Dashboard")
st.markdown("Analyze generalized PPG features and interactively screen subjects for POTS.")

tab1 = st.container()

def extract_features_from_raw(signal, fs=100.0):
    signal = np.array(signal)
    signal = np.nan_to_num(signal)
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    range_val = max_val - min_val
    median_val = np.median(signal)
    var_val = np.var(signal)
    
    peaks, _ = find_peaks(signal, distance=fs/2.5) 
    num_peaks = len(peaks)
    
    if num_peaks > 1:
        nn_intervals_samples = np.diff(peaks)
        hr = fs / np.mean(nn_intervals_samples) if np.mean(nn_intervals_samples) > 0 else 0
        hrv = np.std(nn_intervals_samples)
    else:
        hr = 0
        hrv = 0
        
    return {
        'mean': mean_val, 'std': std_val, 'max': max_val, 'min': min_val,
        'range': range_val, 'median': median_val, 'var': var_val,
        'num_peaks': num_peaks, 'hr': hr, 'hrv': hrv
    }

with tab1:
    st.header("Patient Feature Screening")
    st.markdown("Upload a raw PPG CSV segment containing values, OR directly upload a structured feature row CSV containing columns like `mean, std, hrv` etc.")
    
    if model is None:
        st.warning("⚠️ FLAML Model not found. Run `train_flaml.py`.")
        
    uploaded_file = st.file_uploader("Upload CSV (Raw Signal OR Extracted Features)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            cols = [c.lower() for c in df_upload.columns]
            
            required_features = ['mean', 'std', 'max', 'min', 'range', 'median', 'var', 'num_peaks', 'hr', 'hrv']
            if all(f in cols for f in required_features):
                st.success("Detected structured feature dataset!")
                row_idx = st.number_input("Select Row Index to evaluate", min_value=0, max_value=len(df_upload)-1, value=0)
                
                feat = df_upload.iloc[row_idx].to_dict()
                feat = {k.lower(): v for k, v in feat.items()}
                
                st.subheader("Extracted Parameters")
                st.json({k: round(v, 4) for k, v in feat.items() if k in required_features})
                
                if st.button("Predict POTS Risk"):
                    if model is not None:
                        is_pots, prob = predict_pots(model, feat)
                        st.subheader("🤖 Prediction")
                        if is_pots:
                            st.error(f"**HIGH RISK FOR POTS**")
                        else:
                            st.success(f"**LOW RISK / NORMAL**")
                        
                        st.markdown("---")
                        st.subheader("📈 Patient Trend Analysis & Cohort Comparison")
                        if data_df is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(visuals.plot_patient_comparison(data_df, feat.get('hr', 0), 'hr'), use_container_width=True)
                            with col2:
                                st.plotly_chart(visuals.plot_patient_comparison(data_df, feat.get('hrv', 0), 'hrv'), use_container_width=True)

                        st.markdown("---")
                        st.subheader("📄 Clinical Report")
                        pdf_bytes = generate_pots_report(feat, is_pots, prob, signal=None)
                        st.download_button(
                            label="Download Diagnostic Report",
                            data=pdf_bytes,
                            file_name="POTS_Diagnostic_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            else:
                st.info("Processing as Single Patient Data. Ensure the column containing the signal is selected.")
                columns = list(df_upload.columns)
                
                default_idx = 0
                for i, col in enumerate(columns):
                    col_lower = str(col).lower()
                    if 'ppg' in col_lower or 'pleth' in col_lower or 'signal' in col_lower or 'ch' in col_lower:
                        default_idx = i
                        break
                        
                ppg_col = st.selectbox("Select PPG/Signal Column", columns, index=default_idx)
                
                if st.button("Process & Screen"):
                    signal = df_upload[ppg_col].values
                    with st.spinner("Extracting features..."):
                        feat = extract_features_from_raw(signal)
                        st.subheader("Extracted Parameters")
                        st.json({k: round(v, 4) for k, v in feat.items()})
                        
                        if model is not None:
                            is_pots, prob = predict_pots(model, feat)
                            st.subheader("🤖 Prediction")
                            if is_pots:
                                st.error(f"**HIGH RISK FOR POTS**")
                            else:
                                st.success(f"**LOW RISK / NORMAL**")
                                
                            st.markdown("---")
                            st.subheader("📈 Patient Trend Analysis & Signal Visuals")
                            st.plotly_chart(visuals.plot_ppg_signal_with_peaks(signal), use_container_width=True)
                            
                            if data_df is not None:
                                st.markdown("##### Cohort Comparison")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(visuals.plot_patient_comparison(data_df, feat.get('hr', 0), 'hr'), use_container_width=True)
                                with col2:
                                    st.plotly_chart(visuals.plot_patient_comparison(data_df, feat.get('hrv', 0), 'hrv'), use_container_width=True)

                            st.markdown("---")
                            st.subheader("📄 Clinical Report")
                            pdf_bytes = generate_pots_report(feat, is_pots, prob, signal=signal)
                            st.download_button(
                                label="Download Diagnostic Report",
                                data=pdf_bytes,
                                file_name="POTS_Diagnostic_Report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                                
        except Exception as e:
            st.error(f"Error processing file: {e}")


