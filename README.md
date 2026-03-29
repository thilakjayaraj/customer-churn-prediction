# 🛡️ ChurnGuard — AI Customer Churn Prediction

A production-ready **Streamlit** web application that predicts whether a telecom customer will churn, powered by a **RandomForest** machine-learning pipeline.

## ✨ Features

| Tab | Description |
|-----|-------------|
| **Single Prediction** | Fill in one customer's details and get an instant churn prediction with probability |
| **Batch Prediction** | Upload a CSV of customers and download predictions for all rows |
| **Insights Dashboard** | Interactive Plotly charts — churn distribution, feature importance, business insights |
| **About** | Project overview, tech stack, and team credits |

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Scikit-learn (RandomForestClassifier Pipeline)
- Pandas / NumPy
- Plotly (interactive charts)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (one-time)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## 📂 Project Structure

```
customer-churn-prediction/
├── app.py                # Streamlit application
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── README.md
├── models/
│   └── churn_pipeline.pkl
├── assets/
│   ├── feature_importance.csv
│   ├── churn_distribution.csv
│   ├── column_meta.pkl
│   └── metrics.pkl
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push your code to a **GitHub** repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub → click **New app**.
4. Select the repo, branch `main`, and set main file path to `app.py`.
5. Click **Deploy** — your app will be live in ~2 minutes.

> **Tip:** Make sure `models/churn_pipeline.pkl` and the `assets/` folder are committed to the repo.

## 👥 Team

Built for Mini-Project by **Thilak Jayaraj & Team**.

---

*Model accuracy: ~80% | Dataset: Telco Customer Churn (Kaggle)*
