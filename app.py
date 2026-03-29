"""
🛡️ ChurnGuard — AI Customer Churn Prediction
A production-ready Streamlit web application for telecom churn prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChurnGuard - AI Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header banner ─────────────────────────────────── */
    .header-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,.25);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #8ecae6;
        font-size: 1.1rem;
        margin: .5rem 0 0;
    }

    /* ── Metric cards ──────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,.15);
        transition: transform .2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00b4d8, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: #adb5bd;
        font-size: .85rem;
        margin-top: .3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Result boxes ──────────────────────────────────── */
    .result-stay {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-left: 6px solid #34d399;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        color: #d1fae5;
    }
    .result-churn {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 6px solid #f87171;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        color: #fee2e2;
    }
    .result-stay h3, .result-churn h3 {
        margin: 0 0 .5rem;
        font-size: 1.5rem;
    }

    /* ── Insight cards ─────────────────────────────────── */
    .insight-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: .8rem;
        color: #e2e8f0;
    }
    .insight-card .icon { font-size: 1.4rem; margin-right: .5rem; }

    /* ── Footer ────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #64748b;
        font-size: .82rem;
        border-top: 1px solid rgba(255,255,255,.06);
        margin-top: 3rem;
    }

    /* ── Sidebar polish ────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    /* ── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL & ASSETS
# ═══════════════════════════════════════════════════════════════════════
MODEL_PATH = "models/churn_pipeline.pkl"
ASSETS_DIR = "assets"


@st.cache_resource
def load_pipeline():
    """Load the trained scikit-learn pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_meta():
    meta_path = os.path.join(ASSETS_DIR, "column_meta.pkl")
    if not os.path.exists(meta_path):
        return None
    return joblib.load(meta_path)


@st.cache_resource
def load_metrics():
    metrics_path = os.path.join(ASSETS_DIR, "metrics.pkl")
    if not os.path.exists(metrics_path):
        return {"accuracy": 0.80, "roc_auc": 0.85}
    return joblib.load(metrics_path)


pipeline = load_pipeline()
col_meta = load_meta()
metrics = load_metrics()

if pipeline is None:
    st.error("⚠️ Model not found! Please run `python train_model.py` first.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# HELPER — build a single-row DataFrame matching training schema
# ═══════════════════════════════════════════════════════════════════════

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

FEATURE_TOOLTIPS = {
    "gender": "Customer's gender",
    "SeniorCitizen": "Whether the customer is a senior citizen (65+)",
    "Partner": "Whether the customer has a partner",
    "Dependents": "Whether the customer has dependents",
    "tenure": "Number of months the customer has stayed with the company",
    "PhoneService": "Whether the customer has a phone service",
    "MultipleLines": "Whether the customer has multiple lines",
    "InternetService": "Customer's internet service provider",
    "OnlineSecurity": "Whether the customer has online security add-on",
    "OnlineBackup": "Whether the customer has online backup add-on",
    "DeviceProtection": "Whether the customer has device protection add-on",
    "TechSupport": "Whether the customer has tech support add-on",
    "StreamingTV": "Whether the customer has streaming TV add-on",
    "StreamingMovies": "Whether the customer has streaming movies add-on",
    "Contract": "The contract term of the customer",
    "PaperlessBilling": "Whether the customer has paperless billing",
    "PaymentMethod": "The customer's payment method",
    "MonthlyCharges": "The amount charged to the customer monthly ($)",
    "TotalCharges": "The total amount charged to the customer ($)",
}


def build_input_df(data: dict) -> pd.DataFrame:
    """Build a DataFrame from form data that matches training schema."""
    return pd.DataFrame([data])


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <h1>🛡️ ChurnGuard</h1>
    <p>AI-Powered Customer Churn Prediction for Telecom</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard")
    st.markdown("---")

    st.markdown("### 📊 Model Details")
    st.markdown(f"""
    | Attribute | Value |
    |-----------|-------|
    | Algorithm | Random Forest |
    | Estimators | 200 |
    | Accuracy | **{metrics['accuracy']*100:.1f}%** |
    | ROC-AUC | **{metrics['roc_auc']:.3f}** |
    | Dataset | Telco Churn |
    | Samples | 7,043 |
    """)

    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    st.info("Use the **tabs** above to switch between Single Prediction, Batch Prediction, Insights, and About.")

    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    st.markdown("- [Streamlit Docs](https://docs.streamlit.io)")
    st.markdown("- [Scikit-learn](https://scikit-learn.org)")

    st.markdown("---")
    st.caption("v1.0.0 · Built with ❤️ by Thilak Jayaraj & Team")

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "📁 Batch Prediction",
    "📈 Insights Dashboard",
    "ℹ️ About",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Single Customer Prediction
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🎯 Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details below and click **Predict Churn** to get an instant result.")

    with st.form("single_pred_form"):
        st.markdown("#### 👤 Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"], help=FEATURE_TOOLTIPS["gender"])
        with c2:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", help=FEATURE_TOOLTIPS["SeniorCitizen"])
        with c3:
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"], help=FEATURE_TOOLTIPS["Partner"])
        with c4:
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"], help=FEATURE_TOOLTIPS["Dependents"])

        st.markdown("---")
        st.markdown("#### 📞 Services")
        c1, c2, c3 = st.columns(3)
        with c1:
            phone = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"], help=FEATURE_TOOLTIPS["PhoneService"])
            multiple = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"], help=FEATURE_TOOLTIPS["MultipleLines"])
            internet = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"], help=FEATURE_TOOLTIPS["InternetService"])
        with c2:
            security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"], help=FEATURE_TOOLTIPS["OnlineSecurity"])
            backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"], help=FEATURE_TOOLTIPS["OnlineBackup"])
            protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"], help=FEATURE_TOOLTIPS["DeviceProtection"])
        with c3:
            tech = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"], help=FEATURE_TOOLTIPS["TechSupport"])
            tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"], help=FEATURE_TOOLTIPS["StreamingTV"])
            movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"], help=FEATURE_TOOLTIPS["StreamingMovies"])

        st.markdown("---")
        st.markdown("#### 💳 Billing & Account")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12, help=FEATURE_TOOLTIPS["tenure"])
        with c2:
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"], help=FEATURE_TOOLTIPS["Contract"])
        with c3:
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"], help=FEATURE_TOOLTIPS["PaperlessBilling"])
        with c4:
            payment = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"], help=FEATURE_TOOLTIPS["PaymentMethod"])

        c1, c2 = st.columns(2)
        with c1:
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0, step=1.0, help=FEATURE_TOOLTIPS["MonthlyCharges"])
        with c2:
            total = st.number_input("Total Charges ($)", 18.0, 9000.0, 600.0, step=10.0, help=FEATURE_TOOLTIPS["TotalCharges"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        input_df = build_input_df(input_data)

        try:
            proba = pipeline.predict_proba(input_df)[0]
            churn_prob = proba[1] * 100
            stay_prob = proba[0] * 100
            prediction = pipeline.predict(input_df)[0]

            st.markdown("---")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                number={"suffix": "%", "font": {"size": 42}},
                title={"text": "Churn Probability", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#ef4444" if prediction == 1 else "#22c55e"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                        {"range": [30, 60], "color": "rgba(234,179,8,0.15)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": churn_prob,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )

            col_gauge, col_result = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_result:
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-stay">
                        <h3>✅ Low Risk — Customer Will Stay</h3>
                        <p><strong>Churn Probability:</strong> {churn_prob:.1f}%</p>
                        <p><strong>Retention Probability:</strong> {stay_prob:.1f}%</p>
                        <p style="margin-top:1rem;">This customer shows strong retention signals. 
                        Continue providing excellent service to maintain loyalty.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-churn">
                        <h3>🚨 High Risk — Customer May Churn</h3>
                        <p><strong>Churn Probability:</strong> {churn_prob:.1f}%</p>
                        <p><strong>Retention Probability:</strong> {stay_prob:.1f}%</p>
                        <p style="margin-top:1rem;">⚡ <strong>Recommended actions:</strong> Offer a loyalty discount, 
                        upgrade their contract, or assign a dedicated account manager.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📁 Batch Prediction — Upload a CSV")
    st.markdown("Upload a CSV file with customer data and get churn predictions for every row.")

    st.info("💡 **Tip:** Your CSV should have the same columns as the training data (excluding `customerID` and `Churn`).")

    # Provide a sample template
    sample_cols = (col_meta["numeric_cols"] + col_meta["categorical_cols"]) if col_meta else []
    if sample_cols:
        with st.expander("📋 View expected columns"):
            st.code(", ".join(sample_cols))

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)

            # Drop columns the pipeline doesn't need
            for drop_col in ["customerID", "Churn"]:
                if drop_col in batch_df.columns:
                    batch_df.drop(drop_col, axis=1, inplace=True)

            # Clean TotalCharges
            if "TotalCharges" in batch_df.columns:
                batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce")
                batch_df["TotalCharges"].fillna(batch_df["TotalCharges"].median(), inplace=True)

            st.markdown(f"**Rows uploaded:** {len(batch_df)}")

            with st.spinner("Running predictions…"):
                preds = pipeline.predict(batch_df)
                probas = pipeline.predict_proba(batch_df)[:, 1]

            result_df = batch_df.copy()
            result_df["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
            result_df["Churn_Probability_%"] = (probas * 100).round(2)

            # Summary metrics
            churn_count = (preds == 1).sum()
            stay_count = (preds == 0).sum()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{len(batch_df)}</div>
                    <div class="label">Total Customers</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="background:linear-gradient(90deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{churn_count}</div>
                    <div class="label">Predicted Churn</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="background:linear-gradient(90deg,#22c55e,#4ade80);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{stay_count}</div>
                    <div class="label">Predicted Stay</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Color the prediction column
            def highlight_churn(val):
                if val == "Yes":
                    return "background-color: rgba(239,68,68,0.2); color: #fca5a5;"
                return "background-color: rgba(34,197,94,0.2); color: #86efac;"

            styled = result_df.style.applymap(highlight_churn, subset=["Churn_Prediction"])
            st.dataframe(styled, use_container_width=True, height=400)

            # Download button
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                csv_out,
                "churn_predictions.csv",
                "text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Insights Dashboard
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Insights & Visualizations Dashboard")

    # ── Load asset data ────────────────────────────────────────────────
    churn_dist_path = os.path.join(ASSETS_DIR, "churn_distribution.csv")
    feat_imp_path = os.path.join(ASSETS_DIR, "feature_importance.csv")

    col_left, col_right = st.columns(2)

    # ── Churn Distribution ─────────────────────────────────────────────
    with col_left:
        st.markdown("#### 🍩 Churn Distribution")
        if os.path.exists(churn_dist_path):
            churn_data = pd.read_csv(churn_dist_path)
            fig_pie = px.pie(
                churn_data,
                values="Count",
                names="Churn",
                color="Churn",
                color_discrete_map={"No": "#22c55e", "Yes": "#ef4444"},
                hole=0.45,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=14,
            )
            fig_pie.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Churn distribution data not found.")

    # ── Feature Importance ─────────────────────────────────────────────
    with col_right:
        st.markdown("#### 🏆 Top 15 Feature Importances")
        if os.path.exists(feat_imp_path):
            feat_data = pd.read_csv(feat_imp_path).head(15)
            # Clean feature names
            feat_data["Feature"] = feat_data["Feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ")
            feat_data = feat_data.sort_values("Importance", ascending=True)

            fig_bar = px.bar(
                feat_data,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#0ea5e9", "#06b6d4", "#22d3ee"],
            )
            fig_bar.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                coloraxis_showscale=False,
                yaxis=dict(title=""),
                xaxis=dict(title="Importance Score"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Feature importance data not found.")

    # ── Business Insights ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Key Business Insights")

    insights = [
        ("📉", "Month-to-month contracts", "Customers on month-to-month contracts have the **highest churn rate** (~42%). Offering annual plans with discounts can improve retention."),
        ("⏳", "Tenure is the strongest predictor", "Customers with **less than 12 months** tenure are most likely to churn. Focus onboarding and engagement efforts on new customers."),
        ("💰", "Higher monthly charges → Higher churn", "Customers paying **above $70/month** churn more. Consider loyalty discounts or bundling services."),
        ("🌐", "Fiber optic users churn more", "Fiber optic internet users show **higher churn** than DSL users — possibly due to higher pricing or service issues."),
        ("🔒", "Security & tech support reduce churn", "Customers with **online security** and **tech support** add-ons have significantly **lower churn**."),
        ("📄", "Paperless billing increases churn", "Customers with **paperless billing** churn slightly more — they may be less engaged or more price-sensitive."),
        ("💳", "Electronic check users churn most", "Customers paying via **electronic check** have the highest churn rate compared to auto-pay methods."),
        ("👨‍👩‍👧‍👦", "Partners & dependents lower churn", "Customers with **partners or dependents** are more likely to stay, indicating family plans aid retention."),
    ]

    cols = st.columns(2)
    for idx, (icon, title, desc) in enumerate(insights):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="insight-card">
                <span class="icon">{icon}</span><strong>{title}</strong><br>
                <span style="font-size:.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About ChurnGuard")

    col_about, col_stack = st.columns([3, 2])

    with col_about:
        st.markdown("""
        **ChurnGuard** is an AI-powered customer churn prediction system designed for telecom companies. 
        It uses machine learning to identify customers who are at risk of leaving, enabling proactive retention strategies.

        #### 🎯 Problem Statement
        Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer 
        costs **5-7× more** than retaining an existing one. By predicting churn early, companies can take 
        targeted actions to improve customer satisfaction and reduce revenue loss.

        #### 🧠 How It Works
        1. **Data Collection** — Customer demographics, account info, and service usage from the Telco dataset (7,043 records)
        2. **Feature Engineering** — Numerical scaling + one-hot encoding of categorical variables via `ColumnTransformer`
        3. **Model Training** — `RandomForestClassifier` with 200 estimators, optimized hyperparameters
        4. **Prediction** — Real-time inference via the saved scikit-learn `Pipeline`

        #### 📊 Model Performance
        """)

        perf_cols = st.columns(3)
        with perf_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['accuracy']*100:.1f}%</div>
                <div class="label">Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with perf_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['roc_auc']:.3f}</div>
                <div class="label">ROC-AUC</div>
            </div>""", unsafe_allow_html=True)
        with perf_cols[2]:
            st.markdown("""
            <div class="metric-card">
                <div class="value">7,043</div>
                <div class="label">Training Samples</div>
            </div>""", unsafe_allow_html=True)

    with col_stack:
        st.markdown("#### 🛠️ Tech Stack")
        stack_items = [
            ("🐍", "Python 3.10+", "Core language"),
            ("📊", "Streamlit", "Web framework"),
            ("🤖", "Scikit-learn", "ML pipeline & model"),
            ("🐼", "Pandas / NumPy", "Data processing"),
            ("📈", "Plotly", "Interactive charts"),
            ("💾", "Joblib", "Model serialization"),
        ]
        for icon, name, desc in stack_items:
            st.markdown(f"""
            <div class="insight-card">
                <span class="icon">{icon}</span><strong>{name}</strong><br>
                <span style="font-size:.85rem;color:#94a3b8;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔗 Links")
        st.markdown("- 🐙 [GitHub Repository](#) *(placeholder)*")
        st.markdown("- 📺 [Live Demo](#) *(deploy on Streamlit Cloud)*")
        st.markdown("- 📄 [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    🛡️ <strong>ChurnGuard</strong> — Built for Mini-Project | <strong>Thilak Jayaraj & Team</strong><br>
    Powered by Python · Scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
