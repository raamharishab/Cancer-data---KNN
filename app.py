"""
app.py – Cancer Diagnosis Predictor (KNN)
Run with:  python3 -m streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Diagnosis Predictor",
    page_icon="🔬",
    layout="centered",
)

# ── Injected CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #0a0a1a !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1a0533 0%, #0a0a1a 45%),
                radial-gradient(ellipse at 80% 80%, #04152d 0%, #0a0a1a 55%) !important;
    background-blend-mode: screen !important;
    min-height: 100vh;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(200,160,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 60%, rgba(160,200,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 50% 30%, rgba(255,200,160,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 75%, rgba(160,255,200,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 20%, rgba(255,160,200,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 85%, rgba(200,200,255,0.3) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 65% 50%, rgba(255,255,200,0.25) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 820px !important; }

/* Hero */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-icon {
    font-size: 3.8rem; line-height: 1; margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 24px rgba(168,85,247,0.8));
    animation: pulse-icon 3s ease-in-out infinite;
}
@keyframes pulse-icon {
    0%, 100% { filter: drop-shadow(0 0 24px rgba(168,85,247,0.8)); transform: scale(1); }
    50%       { filter: drop-shadow(0 0 40px rgba(99,102,241,0.9)); transform: scale(1.06); }
}
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem; letter-spacing: -0.5px;
}
.hero-sub { color: rgba(200,200,230,0.65); font-size: 1rem; max-width: 560px; margin: 0 auto; line-height: 1.6; }

/* Section label */
.section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(168,85,247,0.8); margin: 2rem 0 1rem;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 1.8rem 2rem;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.2rem;
}

/* Widgets */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
}
label, .stSelectbox label, .stNumberInput label {
    color: rgba(200,200,230,0.8) !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
}

/* Predict button */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #0ea5e9 100%) !important;
    color: #fff !important; font-weight: 700 !important; font-size: 1.05rem !important;
    border: none !important; border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
    margin-top: 0.5rem;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.65) !important;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.4); border-radius: 18px;
    padding: 1.8rem 2rem; text-align: center;
    animation: slide-up 0.4s ease; box-shadow: 0 0 40px rgba(239,68,68,0.12);
}
.result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
    border: 1px solid rgba(34,197,94,0.4); border-radius: 18px;
    padding: 1.8rem 2rem; text-align: center;
    animation: slide-up 0.4s ease; box-shadow: 0 0 40px rgba(34,197,94,0.12);
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-emoji { font-size: 3.2rem; margin-bottom: 0.5rem; }
.result-title-pos { font-size: 1.6rem; font-weight: 800; color: #f87171; margin: 0.3rem 0; }
.result-title-neg { font-size: 1.6rem; font-weight: 800; color: #4ade80; margin: 0.3rem 0; }
.result-conf { font-size: 1.05rem; color: rgba(200,200,230,0.7); margin-top: 0.3rem; }

/* Probability bars */
.prob-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.5rem 0; }
.prob-label { color: rgba(200,200,230,0.7); font-size: 0.85rem; min-width: 110px; }
.prob-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 99px; }
.prob-val { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; min-width: 42px; text-align: right; }

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), rgba(99,102,241,0.5), transparent);
    margin: 1.5rem 0; border: none;
}

/* Footer */
.footer { text-align: center; color: rgba(200,200,230,0.3); font-size: 0.78rem; margin-top: 2rem; padding-bottom: 1rem; }
.badge {
    display: inline-block; background: rgba(168,85,247,0.15);
    border: 1px solid rgba(168,85,247,0.3); color: rgba(200,160,255,0.8);
    border-radius: 99px; padding: 0.2rem 0.75rem;
    font-size: 0.72rem; font-weight: 600; margin: 0 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler (train on-the-fly if pkl files are missing) ───────────
@st.cache_resource
def load_artifacts():
    import os, pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    model_path  = "knn_model.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        # Auto-train from CSV (used on Streamlit Cloud where pkl files aren't committed)
        df = pd.read_csv("cancer_data.csv")
        X  = df.drop("diagnosis", axis=1)
        y  = df["diagnosis"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Save for future runs (local)
        try:
            joblib.dump(model,  model_path)
            joblib.dump(scaler, scaler_path)
        except Exception:
            pass   # read-only filesystem on cloud — that's fine

    return model, scaler

model, scaler = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🔬</div>
    <div class="hero-title">Cancer Diagnosis Predictor</div>
    <div class="hero-sub">Enter all patient health metrics below. Our K-Nearest Neighbours model will predict the cancer diagnosis risk with ~91% accuracy.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Patient Demographics</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=40, step=1,
        help="Patient's age in years (1 – 120)"
    )
with col2:
    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "♀ Female" if x == 0 else "♂ Male",
        help="0 = Female, 1 = Male"
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">📋 Clinical & Lifestyle Details</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    bmi = st.number_input(
        "BMI",
        min_value=10.0, max_value=50.0, value=25.0, step=0.1,
        help="Body Mass Index (10 – 50)"
    )
    smoking = st.selectbox(
        "Smoking",
        options=[0, 1],
        format_func=lambda x: "🚬 Smoker" if x else "✅ Non-Smoker",
        help="Does the patient smoke?"
    )
    genetic_risk = st.selectbox(
        "Genetic Risk",
        options=[0, 1, 2],
        format_func=lambda x: ["🟢 Low (0)", "🟡 Medium (1)", "🔴 High (2)"][x],
        help="Genetic predisposition level"
    )
with col4:
    physical_activity = st.number_input(
        "Physical Activity (hrs/week)",
        min_value=0.0, max_value=10.0, value=5.0, step=0.1,
        help="Weekly physical activity in hours (0 – 10)"
    )
    alcohol_intake = st.number_input(
        "Alcohol Intake (units/week)",
        min_value=0.0, max_value=5.0, value=2.0, step=0.1,
        help="Weekly alcohol intake in units (0 – 5)"
    )
    cancer_history = st.selectbox(
        "Personal Cancer History",
        options=[0, 1],
        format_func=lambda x: "⚠️ Yes" if x else "✅ No",
        help="Has the patient previously had cancer?"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Diagnosis", use_container_width=True)

if predict_clicked:
    # Feature order must match training: age, gender, bmi, smoking, genetic_risk,
    # physical_activity, alcohol_intake, cancer_history
    features = np.array([[age, gender, bmi, smoking, genetic_risk,
                          physical_activity, alcohol_intake, cancer_history]])
    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]

    p_negative = probability[0] * 100
    p_positive = probability[1] * 100

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-emoji">⚠️</div>
            <div class="result-title-pos">Cancer Detected</div>
            <div class="result-conf">The model predicts a <strong>positive</strong> diagnosis
            with <strong>{p_positive:.1f}%</strong> confidence.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-emoji">✅</div>
            <div class="result-title-neg">No Cancer Detected</div>
            <div class="result-conf">The model predicts a <strong>negative</strong> diagnosis
            with <strong>{p_negative:.1f}%</strong> confidence.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📈 Probability Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prob-row">
        <span class="prob-label">🟢 No Cancer</span>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{p_negative:.1f}%;background:linear-gradient(90deg,#22c55e,#4ade80);"></div>
        </div>
        <span class="prob-val">{p_negative:.1f}%</span>
    </div>
    <div class="prob-row">
        <span class="prob-label">🔴 Cancer</span>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{p_positive:.1f}%;background:linear-gradient(90deg,#ef4444,#f87171);"></div>
        </div>
        <span class="prob-val">{p_positive:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <span class="badge">KNN · k=5</span>
    <span class="badge">Accuracy ≈ 91%</span>
    <span class="badge">8 Features · 1500 rows</span>
    <br><br>
    For educational purposes only. Not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
