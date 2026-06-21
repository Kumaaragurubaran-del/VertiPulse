import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_feature_importance(automl):
    try:
        estimator = automl.model.estimator
        if hasattr(estimator, 'feature_importances_'):
            importance = estimator.feature_importances_
            features = automl.feature_names_in_
            
            df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
            df_imp = df_imp.sort_values(by='Importance', ascending=True)
            
            fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                         title='Model Feature Importance',
                         color='Importance', color_continuous_scale='Blues')
            fig.update_layout(template="plotly_white")
            return fig
    except Exception as e:
        pass
        
    fig = go.Figure()
    fig.add_annotation(text="Feature importances not available.",
                       xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title='Model Feature Importance')
    return fig

def plot_3d_scatter(df):
    if 'hr' in df.columns and 'hrv' in df.columns and 'var' in df.columns:
        fig = px.scatter_3d(df, x='hr', y='hrv', z='var',
                            color='label',
                            title="Interactive 3D Feature Distribution",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            opacity=0.7)
        fig.update_layout(scene=dict(
            xaxis_title='Heart Rate',
            yaxis_title='HRV',
            zaxis_title='Variance'
        ), margin=dict(l=0, r=0, b=0, t=30))
        return fig
    return go.Figure()

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Magma",
                    title="Feature Correlation Matrix")
    return fig

def plot_hr_distribution(df):
    fig = px.histogram(df, x="hr", color="label", marginal="box",
                       barmode="overlay", title="Heart Rate Overlay by Label")
    fig.update_layout(template="plotly_white")
    return fig

def plot_trend_analysis(df, feature='hr'):
    df_trend = df.reset_index().copy()
    
    df_trend[f'{feature}_smooth'] = df_trend.groupby('label')[feature].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df_trend['Class'] = df_trend['label'].map({0: 'Normal', 1: 'Abnormal (POTS Risk)'})
    
    fig = px.line(df_trend, x='index', y=f'{feature}_smooth', color='Class',
                  title=f"Cohort Trend: {feature.upper()} (50-Sample Moving Average)",
                  labels={'index': 'Sample Index', f'{feature}_smooth': feature.upper()},
                  line_shape="spline", render_mode="svg",
                  color_discrete_map={'Normal': '#1f77b4', 'Abnormal (POTS Risk)': '#d62728'})
    
    fig.update_layout(template="plotly_dark")
    return fig

def plot_ppg_signal_with_peaks(signal, fs=100.0):
    from scipy.signal import find_peaks
    import pandas as pd
    import plotly.graph_objects as go
    
    signal = np.array(signal)
    signal = np.nan_to_num(signal)
    peaks, _ = find_peaks(signal, distance=fs/2.5)
    
    # Calculate trend (Moving Average)
    trend = pd.Series(signal).rolling(window=int(fs), min_periods=1).mean().values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Raw PPG', opacity=0.6, line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(y=trend, mode='lines', name='Signal Trend (1s MA)', line=dict(color='#ff7f0e', dash='dot')))
    
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=peaks, y=signal[peaks], mode='markers', name='Detected Peaks', marker={'color': 'red', 'size': 8, 'symbol': 'x'}))
        
    fig.update_layout(title="Patient Raw PPG Signal & Trend Analysis",
                      xaxis_title="Time (Samples)",
                      yaxis_title="Amplitude",
                      template="plotly_white")
    return fig

def plot_patient_comparison(df, patient_val, feature):
    import plotly.express as px
    
    # Ensure label is categorical for coloring
    df_plot = df.copy()
    df_plot['Class'] = df_plot['label'].map({0: 'Normal', 1: 'Abnormal (POTS)'})
    
    fig = px.histogram(df_plot, x=feature, color="Class", marginal="box",
                       barmode="overlay", title=f"Patient {feature.upper()} vs Cohort Distribution",
                       color_discrete_map={'Normal': '#1f77b4', 'Abnormal (POTS)': '#d62728'})
    
    # Add vertical line for the patient
    fig.add_vline(x=patient_val, line_width=4, line_dash="dash", line_color="green", 
                  annotation_text="Current Patient", annotation_position="top right")
    
    fig.update_layout(template="plotly_white")
    return fig

def plot_risk_gauge(probability):
    import plotly.graph_objects as go
    
    val = probability * 100
    color = "#2ca02c" if val < 40 else "#ff7f0e" if val < 70 else "#d62728"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "POTS Risk Indicator<br><span style='font-size:0.8em;color:gray'>Classification Probability</span>", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, 40], 'color': "rgba(44, 160, 44, 0.15)"},
                {'range': [40, 70], 'color': "rgba(255, 127, 14, 0.15)"},
                {'range': [70, 100], 'color': "rgba(214, 39, 40, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': val
            }
        }
    ))
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=300)
    return fig

def plot_feature_radar(feat_dict):
    import plotly.graph_objects as go
    
    features = ['hr', 'hrv', 'std', 'var', 'range']
    vals = [feat_dict.get(f, 0) for f in features]
    
    # Normalize features logically for visualization
    max_vals = {'hr': 200, 'hrv': 25, 'std': 10, 'var': 100, 'range': 20}
    normalized_vals = [min(1.0, max(0.0, vals[i] / max_vals.get(features[i], 1.0))) * 100 for i in range(len(features))]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_vals + [normalized_vals[0]], # Close the loop
        theta=[f.upper() for f in features] + [features[0].upper()],
        fill='toself',
        name='Patient Signature',
        line_color='rgba(138, 43, 226, 0.8)',
        fillcolor='rgba(138, 43, 226, 0.4)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )),
        showlegend=False,
        title="Patient Autonomic Signature Pattern",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig
