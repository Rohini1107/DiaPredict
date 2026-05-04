import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="DiaPredict", layout="wide")

# =========================
# 🎨 PREMIUM CSS
# =========================
st.markdown("""
<style>

/* NAV BAR TEXT */
button[data-baseweb="tab"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #cbd5f5 !important;
}

/* ACTIVE TAB */
button[data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
    border-bottom: 3px solid #a855f7 !important;
}

/* HOVER */
button[data-baseweb="tab"]:hover {
    color: #60a5fa !important;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #020617, #020617, #0f172a);
}

/* BUTTON STYLE */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #9333ea);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    border: none;
}

/* BUTTON HOVER */
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #c026d3);
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
svm = pickle.load(open("svm.pkl", "rb"))
ridge = pickle.load(open("ridge.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

# =========================
# SESSION HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# HEADER
# =========================
st.markdown("<h1 style='color:white;'>🧠 DiaPredict</h1>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Insights", "📜 History", "💡 Tips"])

# =========================================================
# 🏠 DASHBOARD
# =========================================================
with tab1:

    st.subheader("🩺 Enter Health Details")

    c1,c2,c3 = st.columns(3)

    with c1:
        highbp = st.radio("High BP", ["No","Yes"])
        highbp = 1 if highbp=="Yes" else 0
        bmi = st.slider("BMI",10.0,50.0,25.0)

    with c2:
        smoker = st.radio("Smoker", ["No","Yes"])
        smoker = 1 if smoker=="Yes" else 0
        age = st.slider("Age",18,80,30)

    with c3:
        genhlth = st.slider("General Health",1,5,3)
        physhlth = st.slider("Physical Health Days",0,30,5)
        menthlth = st.slider("Mental Health Days",0,30,5)

    # PREP DATA
    data = np.array([[highbp,bmi,smoker,age,genhlth,physhlth,menthlth]])
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🚀 Predict Risk", use_container_width=True)

    with col2:
        forecast_btn = st.button("📅 Forecast 7 Days", use_container_width=True)

    # =========================
    # PREDICT
    # =========================
    if predict_btn:

        pred = svm.predict(data_pca)[0]
        risk = float(ridge.predict(data_pca)[0])
        cluster = int(kmeans.predict(data_pca)[0])

        level = "Low" if risk<0.5 else "Medium" if risk<0.7 else "High"

        st.success(f"Prediction: {level} Risk | Score: {round(risk,2)}")

        # SAVE HISTORY
        st.session_state.history.append({
            "Time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "Type": "Prediction",
            "Risk": round(risk,2),
            "Level": level,
            "Cluster": cluster
        })

        # RISK GRAPH
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[0.2,0.3,0.4,risk],
            mode='lines+markers'
        ))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # FORECAST
    # =========================
    if forecast_btn:

        st.subheader("📅 7-Day Diabetes Risk Forecast")

        base_risk = float(ridge.predict(data_pca)[0])

        future = []
        current = base_risk

        for i in range(7):
            change = np.random.uniform(-0.02, 0.04)
            current = max(0, min(1, current + change))
            future.append(round(current, 2))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(7)],
            y=future,
            mode='lines+markers',
            line=dict(color='#a855f7', width=3)
        ))

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

        st.success(f"📈 Estimated future risk starts from {round(base_risk,2)}")

        # SAVE HISTORY
        st.session_state.history.append({
            "Time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "Type": "Forecast",
            "Risk": round(base_risk,2),
            "Level": "Trend",
            "Cluster": "-"
        })

# =========================================================
# 📊 INSIGHTS
# =========================================================
with tab2:

    st.subheader("📊 Health Insights")

    st.markdown("""
- 📈 Higher BMI increases diabetes risk  
- 🚬 Smoking increases insulin resistance  
- 🧠 Mental stress affects glucose levels  
- 🏃 Exercise reduces diabetes risk  
- ❤️ Heart diseases are linked to diabetes  
""")

# =========================================================
# 📜 HISTORY
# =========================================================
with tab3:

    st.subheader("📜 Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
    else:
        st.info("No history yet")

# =========================================================
# 💡 TIPS
# =========================================================
with tab4:

    st.subheader("💡 Tips to Stay Healthy")

    st.markdown("""
- 🥗 Eat a balanced diet  
- 🏃 Exercise daily  
- 🚭 Avoid smoking  
- 😴 Get proper sleep  
- 💧 Stay hydrated  
- 🩺 Regular checkups  
""")

# =========================
# FOOTER
# =========================
st.write("---")
st.caption("DiaPredict • AI Healthcare Dashboard 🚀")