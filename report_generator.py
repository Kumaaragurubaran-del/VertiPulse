import tempfile
import os
from fpdf import FPDF
import matplotlib.pyplot as plt

def generate_pots_report(patient_features, is_pots, probability, signal=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Set up styling
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, txt="POTS Diagnostic Clinical Report", ln=True, align='C')
    pdf.ln(5)
    
    # Line separator
    pdf.set_draw_color(0, 51, 102)
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)
    
    # Risk Assessment Section
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt="1. Risk Assessment", ln=True)
    
    pdf.set_font("Arial", '', 12)
    risk_text = "HIGH RISK FOR POTS" if is_pots else "LOW RISK / NORMAL"
    # Highlight risk in red if high, green if low
    if is_pots:
        pdf.set_text_color(204, 0, 0)
    else:
        pdf.set_text_color(0, 153, 0)
        
    pdf.cell(0, 10, txt=f"Diagnosis: {risk_text}", ln=True)
    pdf.cell(0, 10, txt=f"Probability: {probability * 100:.1f}%", ln=True)
    pdf.ln(5)
    
    # HRV Statistics Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="2. Physiological Statistics (PPG Derived)", ln=True)
    
    pdf.set_font("Arial", '', 11)
    # create a simple table-like format
    for key, value in patient_features.items():
        if isinstance(value, float):
            pdf.cell(100, 8, txt=f"{key.upper()}:", align='L')
            pdf.cell(0, 8, txt=f"{value:.4f}", align='L', ln=True)
        else:
            pdf.cell(100, 8, txt=f"{key.upper()}:", align='L')
            pdf.cell(0, 8, txt=f"{value}", align='L', ln=True)
            
    pdf.ln(5)
    
    # PPG Visual
    if signal is not None and len(signal) > 0:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="3. PPG Signal Segment", ln=True)
        
        plt.figure(figsize=(10, 3))
        plt.plot(signal[:1000], color='#1f77b4', linewidth=1.5)
        plt.title("Photoplethysmography (PPG) Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name, format="png", dpi=150)
            plt.close()
            
            # Insert image into PDF
            pdf.image(tmpfile.name, x=15, w=180)
            
        try:
            os.remove(tmpfile.name)
        except Exception:
            pass
            
    # Save to bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
            
    try:
        os.remove(tmp.name)
    except Exception:
        pass
        
    return pdf_bytes
