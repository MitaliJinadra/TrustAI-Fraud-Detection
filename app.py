import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# TRUSTAI | PREMIUM FINTECH DASHBOARD
# =========================================================

st.set_page_config(
    page_title="TrustAI | Intelligent Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Visual theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080d1b;
    --panel: #111b31;
    --panel2: #15233d;
    --line: rgba(148, 170, 210, .16);
    --muted: #9aabc8;
    --white: #f4f7ff;
    --blue: #63a7ff;
    --cyan: #55e0dc;
    --purple: #a78bfa;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(ellipse at 75% 0%, rgba(50, 105, 210, .18), transparent 34%),
      radial-gradient(ellipse at 5% 35%, rgba(74, 222, 214, .07), transparent 28%),
      var(--bg);
    color: var(--white);
}
[data-testid="stHeader"] { background: rgba(8,13,27,.88); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b31 0%, #0b1427 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #e8efff; }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebbd2; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--white) !important; }
p, label, .stCaption, [data-testid="stWidgetLabel"] { color: #c0cce0; }
.main-hero {
    position: relative; overflow: hidden;
    padding: 30px 34px; border-radius: 24px;
    background: linear-gradient(115deg, rgba(28,55,99,.96), rgba(20,36,68,.94) 52%, rgba(30,32,76,.96));
    border: 1px solid rgba(117,166,255,.28);
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
    margin-bottom: 24px;
}
.main-hero:after {
    content: ""; position: absolute; width: 260px; height: 260px;
    right: -55px; top: -120px; border-radius: 50%;
    background: radial-gradient(circle, rgba(85,224,220,.28), transparent 68%);
}
.eyebrow { color: #8de9e4; letter-spacing: 2.5px; font-size: 11px; font-weight: 800; }
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-size: clamp(30px, 4vw, 46px);
    font-weight: 700; line-height: 1.1; margin: 10px 0 8px;
    background: linear-gradient(90deg, #ffffff, #8fc5ff 55%, #72e4df);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #c0d2ee; font-size: 15px; margin: 0; }
.hero-pill {
    display: inline-block; margin-top: 18px; padding: 7px 12px;
    border: 1px solid rgba(141,233,228,.35); border-radius: 30px;
    color: #b8f6ef; background: rgba(85,224,220,.08); font-size: 12px; font-weight: 700;
}
.section-heading {
    font-family: 'Space Grotesk', sans-serif; font-size: 21px;
    font-weight: 700; color: #eaf2ff; margin: 26px 0 13px;
}
.section-kicker { color: #7fb8ff; font-size: 11px; font-weight: 800; letter-spacing: 2px; }
.metric-card {
    background: linear-gradient(145deg, rgba(23,39,67,.98), rgba(15,27,49,.98));
    border: 1px solid rgba(126,165,225,.2); border-radius: 18px;
    padding: 20px 21px; min-height: 128px;
    box-shadow: 0 10px 30px rgba(0,0,0,.12);
}
.metric-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.metric-label { color:#aabbd5; font-size:13px; font-weight:600; }
.metric-icon { font-size:19px; }
.metric-value { color:#f7faff; font-family:'Space Grotesk',sans-serif; font-size:31px; font-weight:700; margin-top:12px; }
.metric-note { color:#8297b8; font-size:11px; margin-top:2px; }
.info-panel {
    background: rgba(18,32,57,.82); border:1px solid var(--line);
    border-radius:16px; padding:17px 19px; color:#c7d5eb;
}
.step-card {
    background: rgba(17,29,51,.72); border:1px solid var(--line);
    border-radius:18px; padding:18px 20px; margin: 8px 0 14px;
}
.stButton > button, .stDownloadButton > button {
    border: 0 !important; border-radius: 11px !important;
    background: linear-gradient(100deg, #2877e8, #4a9cff) !important;
    color: white !important; font-weight: 800 !important;
    min-height: 46px; box-shadow: 0 8px 22px rgba(40,119,232,.22);
    transition: transform .18s ease, filter .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.12); transform: translateY(-1px);
}
.stSelectbox > div > div, .stFileUploader section {
    background: #14233d !important; border: 1px solid #314766 !important;
    border-radius: 12px !important; color: #f4f7ff !important;
}
.stFileUploader label, .stSelectbox label { color:#dce8fa !important; font-weight:600 !important; }
.stFileUploader small { color:#aebed6 !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
.stAlert { border-radius: 13px; }
hr { border-color: var(--line) !important; }
.sidebar-brand { font-family:'Space Grotesk',sans-serif; font-size:25px; font-weight:700; color:#f5f8ff; }
.sidebar-sub { color:#8fa5c5; font-size:12px; margin-top:3px; }
.sidebar-status {
    padding:12px 13px; border-radius:12px; margin-top:8px;
    background:rgba(47,190,143,.1); border:1px solid rgba(47,190,143,.25);
    color:#8be4bd; font-size:12px; font-weight:700;
}
.footer { color:#7085a6; font-size:11px; text-align:center; padding:22px 0 4px; }
.small-muted { color:#92a6c5; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Load trained models ----------
@st.cache_resource
def load_models():
    logistic = joblib.load("models/trustai_model.pkl")
    random_forest = joblib.load("models/trustai_random_forest.pkl")
    return logistic, random_forest

try:
    logistic_model, random_forest_model = load_models()
except Exception as error:
    st.error(f"Unable to load the trained models: {error}")
    st.info("Check that both model files are present inside the models folder.")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div style="font-size:32px">🛡️</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand">TrustAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">INTELLIGENT FRAUD ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Dashboard", "About TrustAI"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.markdown('<div class="sidebar-status">● &nbsp; MODELS READY</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="small-muted">Machine learning powered<br>Academic prototype • v1.0</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-muted">Built with Python · Scikit-learn · Streamlit</div>', unsafe_allow_html=True)

expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown("""
    <div class="main-hero">
      <div class="eyebrow">INTELLIGENCE • SECURITY • INSIGHT</div>
      <div class="hero-title">See beyond the transaction.</div>
      <p class="hero-sub">AI-powered credit card transaction analysis to identify patterns that may indicate fraud.</p>
      <span class="hero-pill">✦ TRUSTAI ANALYTICS WORKSPACE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">01 / CONFIGURATION</div><div class="section-heading">Choose your detection engine</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        model_choice = st.selectbox("Machine learning model", ["Random Forest", "Logistic Regression"])
    selected_model = random_forest_model if model_choice == "Random Forest" else logistic_model
    with right:
        st.markdown(f"""
        <div class="info-panel" style="margin-top:27px">
          <span style="color:#7fe5df;font-weight:800">ACTIVE MODEL</span><br>
          <span style="font-family:'Space Grotesk';font-size:19px;font-weight:700;color:#f4f7ff">{model_choice}</span>
          <span style="color:#93a8c7;font-size:12px"> &nbsp; • &nbsp; Ready for analysis</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">02 / DATA INPUT</div><div class="section-heading">Upload transaction data</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Upload a CSV containing the 30 model input fields: Time, V1–V28 and Amount.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a transaction CSV file", type=["csv"], label_visibility="visible")

    if uploaded_file is None:
        st.markdown("""
        <div class="step-card">
          <div style="font-size:26px">📂</div>
          <div style="font-weight:700;color:#eaf2ff;margin:7px 0">Your analysis starts here</div>
          <div class="small-muted">Choose a CSV file above. TrustAI will validate its columns before running predictions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            data = pd.read_csv(uploaded_file)
            # The original dataset label is deliberately excluded from model inputs.
            if "Class" in data.columns:
                data = data.drop(columns=["Class"])

            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                st.error("Missing required columns: " + ", ".join(missing_columns))
            else:
                X = data[expected_columns].copy()
                for col in expected_columns:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

                if X.isnull().values.any():
                    st.error("The file contains missing or non-numeric values in the required model columns.")
                elif not np.isfinite(X.to_numpy(dtype=float)).all():
                    st.error("The file contains infinite or otherwise invalid numeric values.")
                elif len(X) == 0:
                    st.error("The uploaded CSV contains no transaction rows.")
                else:
                    st.markdown('<div class="section-kicker">03 / DATA CHECK</div><div class="section-heading">Dataset preview</div>', unsafe_allow_html=True)
                    p1, p2, p3 = st.columns(3)
                    for col, label, value in [
                        (p1, "ROWS DETECTED", f"{len(data):,}"),
                        (p2, "MODEL INPUTS", "30"),
                        (p3, "FILE TYPE", "CSV")
                    ]:
                        with col:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
                    with st.expander("Preview first 5 transactions", expanded=False):
                        st.dataframe(data.head(5), use_container_width=True)

                    st.markdown('<div class="section-kicker">04 / RUN ANALYSIS</div><div class="section-heading">Ready to inspect transactions?</div>', unsafe_allow_html=True)
                    if st.button("⚡  ANALYSE TRANSACTIONS", use_container_width=True):
                        with st.spinner("TrustAI is analysing the uploaded transactions..."):
                            predictions = selected_model.predict(X)

                        result = data.copy()
                        result["Prediction"] = np.where(predictions == 1, "Predicted Fraud", "Predicted Normal")
                        fraud_count = int((predictions == 1).sum())
                        normal_count = int((predictions == 0).sum())
                        total = len(predictions)
                        fraud_percentage = fraud_count / total * 100

                        st.markdown('<div class="section-kicker">05 / INSIGHT OVERVIEW</div><div class="section-heading">Detection results</div>', unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3, gap="medium")
                        cards = [
                            (m1, "TOTAL TRANSACTIONS", f"{total:,}", "▦", "Rows analysed"),
                            (m2, "PREDICTED NORMAL", f"{normal_count:,}", "✓", "Classified as normal"),
                            (m3, "PREDICTED FRAUD", f"{fraud_count:,}", "⚑", "Flagged for review")
                        ]
                        for col, label, value, icon, note in cards:
                            with col:
                                st.markdown(f"""
                                <div class="metric-card">
                                  <div class="metric-top"><span class="metric-label">{label}</span><span class="metric-icon">{icon}</span></div>
                                  <div class="metric-value">{value}</div>
                                  <div class="metric-note">{note}</div>
                                </div>
                                """, unsafe_allow_html=True)

                        st.markdown("")
                        st.markdown(f"""
                        <div class="info-panel">
                          <span style="color:#8fe9df;font-weight:800">PREDICTED FRAUD RATE</span>
                          <span style="float:right;color:#fff;font-family:'Space Grotesk';font-size:21px;font-weight:700">{fraud_percentage:.3f}%</span>
                          <div style="height:7px;background:#253854;border-radius:9px;margin-top:13px;overflow:hidden">
                            <div style="height:7px;width:{min(fraud_percentage,100):.3f}%;background:linear-gradient(90deg,#4b9cff,#55e0dc);border-radius:9px"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown('<div class="section-kicker">06 / VISUAL ANALYTICS</div><div class="section-heading">Transaction distribution</div>', unsafe_allow_html=True)
                        # A logarithmic scale keeps the small fraud bar visible beside
                        # the much larger normal-transaction count.
                        fig, ax = plt.subplots(figsize=(10, 3.7))
                        fig.patch.set_facecolor("#111b31")
                        ax.set_facecolor("#111b31")
                        categories = ["Predicted normal", "Predicted fraud"]
                        counts = [normal_count, fraud_count]
                        bars = ax.barh(
                            categories,
                            counts,
                            color=["#438fe8", "#55d9d1"],
                            height=0.48
                        )
                        ax.set_xscale("log")
                        ax.set_xlabel("Number of transactions (logarithmic scale)", color="#aebed6", labelpad=10)
                        ax.tick_params(axis="x", colors="#9eb0ca", labelsize=9)
                        ax.tick_params(axis="y", colors="#dce8fa", labelsize=10)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_color("#344762")
                        ax.spines["bottom"].set_color("#344762")
                        ax.grid(axis="x", color="#2a3b56", alpha=.6, linewidth=.7)
                        ax.set_axisbelow(True)
                        for bar, count in zip(bars, counts):
                            ax.text(
                                count * 1.12,
                                bar.get_y() + bar.get_height() / 2,
                                f"{count:,}",
                                va="center",
                                color="#f4f7ff",
                                fontsize=10,
                                fontweight="bold"
                            )
                        ax.set_xlim(1, max(counts) * 2.5)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption("The chart uses a logarithmic scale so both transaction categories remain visible despite their very different counts.")

                        st.markdown('<div class="section-kicker">07 / TRANSACTION REVIEW</div><div class="section-heading">Detailed prediction report</div>', unsafe_allow_html=True)
                        filter_choice = st.selectbox("Show transactions", ["All predictions", "Predicted fraud only", "Predicted normal only"], key="result_filter")
                        if filter_choice == "Predicted fraud only":
                            view = result[result["Prediction"] == "Predicted Fraud"]
                        elif filter_choice == "Predicted normal only":
                            view = result[result["Prediction"] == "Predicted Normal"]
                        else:
                            view = result
                        st.caption(f"Showing {len(view):,} of {len(result):,} transactions")

                        # Keep the on-screen review focused on the most useful
                        # fields; all original columns remain in the full CSV export.
                        display_columns = ["Time", "Amount", "Prediction"]
                        table_view = view[display_columns].head(1000).copy()
                        styled_table = (
                            table_view.style
                            .set_properties(**{
                                "background-color": "#111b31",
                                "color": "#eaf2ff",
                                "border-color": "#2a3b56",
                                "padding": "10px"
                            })
                            .set_table_styles([
                                {
                                    "selector": "th",
                                    "props": [
                                        ("background-color", "#1b3152"),
                                        ("color", "#9ed0ff"),
                                        ("font-weight", "700"),
                                        ("border-color", "#2a3b56")
                                    ]
                                },
                                {
                                    "selector": "td",
                                    "props": [("border-color", "#2a3b56")]
                                }
                            ])
                        )
                        st.dataframe(styled_table, use_container_width=True, height=390)
                        st.caption("For a simpler on-screen review, this table shows Time, Amount and Prediction. The complete report below includes all original transaction features.")
                        if len(view) > 1000:
                            st.caption("For performance, the on-screen table shows the first 1,000 matching rows. The downloaded report contains all rows.")

                        csv_data = result.to_csv(index=False).encode("utf-8")
                        st.markdown(
                            '<div class="section-kicker">08 / EXPORT</div>'
                            '<div class="section-heading">Take your analysis with you</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<div class="info-panel" style="margin-bottom:12px">'
                            '<span style="color:#8fe9df;font-weight:800">COMPLETE REPORT READY</span><br>'
                            '<span style="color:#c7d5eb;font-size:13px">Download all analysed transactions, including the 30 input features and model predictions.</span>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        st.download_button(
                            "⬇  DOWNLOAD COMPLETE PREDICTION REPORT",
                            data=csv_data,
                            file_name="trustai_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.warning("Educational prototype: these are model predictions, not verified fraud findings. Flagged transactions require independent review; do not use this tool for real financial decisions.")

        except Exception as error:
            st.error(f"An error occurred while reading or analysing the file: {error}")

# ---------- About page ----------
elif page == "About TrustAI":
    st.markdown("""
    <div class="main-hero">
      <div class="eyebrow">THE PROJECT BEHIND THE PLATFORM</div>
      <div class="hero-title">About TrustAI</div>
      <p class="hero-sub">A machine-learning educational project exploring transaction patterns and fraud classification.</p>
      <span class="hero-pill">🛡️ LEARN • ANALYSE • BUILD</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">What is TrustAI?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
      TrustAI is an educational machine-learning application designed to analyse credit-card transaction data
      and classify transactions as normal or potentially fraudulent. It demonstrates a complete introductory
      workflow: preparing data, training models, evaluating performance and presenting predictions in an
      interactive dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Machine learning models</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-icon">🌲</div>
          <div style="font-family:'Space Grotesk';font-size:20px;font-weight:700;color:#f5f8ff;margin:10px 0">Random Forest</div>
          <div class="small-muted">An ensemble learning method that combines multiple decision trees to produce a classification.</div>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-icon">📈</div>
          <div style="font-family:'Space Grotesk';font-size:20px;font-weight:700;color:#f5f8ff;margin:10px 0">Logistic Regression</div>
          <div class="small-muted">A classification algorithm used as a baseline model for comparing predictive performance.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Technology stack</div>', unsafe_allow_html=True)
    techs = ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Streamlit", "Joblib"]
    st.markdown(" ".join([f'<span style=\"display:inline-block;padding:8px 13px;margin:4px;border-radius:20px;background:#172b49;border:1px solid #355274;color:#cfe3ff;font-size:12px;font-weight:700\">{t}</span>' for t in techs]), unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Our Mission & Objectives</div>
    <div class="info-panel">
        <h3>Making transaction analysis smarter</h3>
        TrustAI explores how machine learning can help identify unusual
        patterns in credit card transactions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>01 · Analyse</h3>
            <p>Explore transaction data and identify useful patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>02 · Predict</h3>
            <p>Use machine learning models to classify transactions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>03 · Understand</h3>
            <p>Present results through charts and downloadable reports.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">How TrustAI Works</div>
    <div class="info-panel">
        <h3>From data to insights</h3>
        <p>Data Upload → Data Preparation → Model Selection →
        Prediction → Visualisation → Report Export</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Dataset Overview</div>
    <div class="info-panel">
        <h3>Credit Card Transactions Dataset</h3>
        <p><b>284,807</b> original transactions</p>
        <p><b>30</b> model input features: Time, V1–V28 and Amount</p>
        <p>The original dataset also includes a Class label used for
        model training and evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-heading">Important project note</div>', unsafe_allow_html=True)
    st.warning("TrustAI is an academic prototype. Its predictions are not confirmed financial fraud findings and must not be used to make real financial decisions.")

# ---------- Footer ----------
st.markdown('<div class="footer">TRUSTAI  •  INTELLIGENT FRAUD ANALYTICS  •  ACADEMIC PROJECT</div>', unsafe_allow_html=True)
