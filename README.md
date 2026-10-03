# 🛒 Retail Analytics Intelligence Platform

> **An industry-level, end-to-end Data Science project built on a 120,000-transaction supermarket dataset.**  
> Covers EDA, Customer Segmentation, Churn Prediction, Sales Forecasting, Promotion Effectiveness, and an interactive Streamlit dashboard — all in Python.

---

## 📌 Project Description

The **Retail Analytics Intelligence Platform** transforms raw supermarket transaction data into actionable business intelligence through five interconnected analytical modules:

| Module                      | Technique                                     | Business Outcome                                             |
| --------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| Exploratory Data Analysis   | Pandas, Plotly, Seaborn                       | Revenue drivers, seasonal patterns, demographics             |
| Customer Segmentation       | K-Means + RFM Analysis                        | 4 actionable customer tiers for targeted marketing           |
| Churn Prediction            | XGBoost Classifier                            | Identify at-risk customers before they leave                 |
| Sales Forecasting           | Facebook Prophet                              | Daily revenue projections with confidence intervals          |
| Promotion & Margin Analysis | Discount-band and category margin comparisons | Identify discount levels and categories for promotion review |

All insights are surfaced through a **multi-page Streamlit dashboard** with live model inference.

---

## 📂 Dataset

| Property       | Value                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------ |
| **File**       | [`supermarket_large_dataset.csv`](https://www.kaggle.com/datasets/datascikhan/supermarket) |
| **Rows**       | 120,000 transactions                                                                       |
| **Columns**    | 323 features                                                                               |
| **Domain**     | Retail / Supermarket                                                                       |
| **Time Range** | 2023-01-01 to 2024-05-31                                                                   |
| **Geography**  | USA — multi-region, multi-store                                                            |

**Key Feature Domains:** Transaction details · Customer demographics & loyalty · Product information · Store attributes · Geographic data · Supply chain · Financial metrics (revenue, margin, discount, tax)

---

## 🛠️ Tech Stack

### Architecture

> This project uses **Streamlit** as an all-in-one Python web framework.
> Streamlit runs a single Python process that handles both the UI rendering and all data/model logic — there is **no separate backend server, no Flask, and no REST API**.
> All computation (data loading, ML inference, forecasting) happens directly inside `app.py` and `train_model.py`.

```
┌─────────────────────────────────────────────────────┐
│              User's Browser (localhost:8501)        │
└────────────────────┬────────────────────────────────┘
                     │  HTTP (Streamlit WebSocket)
┌────────────────────▼────────────────────────────────┐
│          Streamlit Server  (app.py)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  UI / Pages │  │  Data Layer  │  │ ML Models  │ │
│  │  (Plotly /  │  │  (pandas /   │  │ (XGBoost / │ │
│  │  Streamlit) │  │   numpy)     │  │  Prophet)  │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
         ▲ reads CSV & .pkl files directly
```

### Dashboard / UI Layer

| Library      | Version | Role                                                      |
| ------------ | ------- | --------------------------------------------------------- |
| `streamlit`  | ≥ 1.27  | Web UI framework — renders all pages, widgets, and layout |
| `plotly`     | ≥ 5.15  | Interactive charts (bar, line, scatter, 3-D, gauge, pie)  |
| `matplotlib` | ≥ 3.7   | Static EDA plots inside the notebook                      |
| `seaborn`    | ≥ 0.12  | Statistical heatmaps and distribution plots               |

### Data & ML Layer (runs inside the same Python process)

| Library        | Version | Role                                                  |
| -------------- | ------- | ----------------------------------------------------- |
| `pandas`       | ≥ 2.0   | Data loading, wrangling, aggregation                  |
| `numpy`        | ≥ 1.24  | Numerical operations                                  |
| `scikit-learn` | ≥ 1.3   | Preprocessing, K-Means clustering, evaluation metrics |
| `xgboost`      | ≥ 1.7   | Churn prediction — XGBoost Classifier                 |
| `prophet`      | ≥ 1.1   | Time-series sales forecasting                         |
| `joblib`       | ≥ 1.3   | Model serialisation (save/load `.pkl` files)          |

---

## 📁 Project Structure

```
retail-analytics-platform/
│
├── supermarket_large_dataset.csv      # Source dataset (120k rows × 323 cols)
│
├── app.py                             # ★ Streamlit dashboard — run this to see the project live
├── train_model.py                     # ★ Run this first — trains & saves the churn model
│
├── Student_SupermarketAnalytics.ipynb # Full analysis notebook (EDA + all ML modules)
│
├── requirements.txt                   # Python dependencies
├── Student_ProjectReport.docx         # Full project report (Word)
└── README.md                          # This file
│
│  (generated after running train_model.py)
├── churn_model_xgb.pkl                # Trained XGBoost model
└── churn_imputer.pkl                  # Feature imputer
```

---

## ⚙️ Setup & Run Instructions

### Prerequisites

- Python **3.9+**
- pip (or conda)

### 1. Clone / Download the project

```bash
git clone https://github.com/yourname/retail-analytics-platform.git
cd retail-analytics-platform
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Prophet** requires a C++ compiler. On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.  
> Alternatively, install via conda: `conda install -c conda-forge prophet`

### 4. Run the Jupyter Notebook

```bash
jupyter lab
# or
jupyter notebook
```

Open `Student_SupermarketAnalytics.ipynb` and run all cells sequentially (top → bottom).  
This will:

- Train and save the XGBoost churn model (`churn_model_xgb.pkl`)
- Save all visualisation images

### 4. Train the churn model _(required before opening the dashboard)_

```bash
python train_model.py
```

> This saves `churn_model_xgb.pkl` and `churn_imputer.pkl` — the dashboard's Churn Predictor page needs these.

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at **`http://localhost:8501`** in your browser.
No Flask, no server setup, no API keys — Streamlit handles everything.

---

## 🗺️ Analytical Modules — Quick Reference

### Module 1: EDA

- Revenue time-series, category breakdown, hourly/weekday patterns
- Customer demographic distributions (age, gender, segment)
- Correlation heatmap of key numeric features

### Module 2: Customer Segmentation

```
RFM Features → Min-Max Scaling → K-Means (K=4) → Business Labels
Champions | Loyal Customers | At-Risk Customers | Lost Customers
```

### Module 3: Churn Prediction

```
21 features → Median Imputation → XGBoost (n=300, lr=0.05)
→ ROC-AUC > 0.85 | Precision/Recall optimised for churn class
```

### Module 4: Sales Forecasting

```
Daily Revenue → Prophet (multiplicative, yearly+weekly seasonality)
→ 60-day forward forecast with 95% confidence intervals
```

### Module 5: Promotion Effectiveness & Gross-Margin Analysis

```
Discount bands and promotion status → gross sales, gross profit, weighted margin
→ Lowest-margin product categories prioritized for promotion review
```

---

## 📊 Key Results

| Metric                        | Value                      |
| ----------------------------- | -------------------------- |
| XGBoost ROC-AUC               | > 0.85                     |
| Segmentation Silhouette Score | > 0.45                     |
| Champion Customers (% of CLV) | ~60%                       |
| Predicted Churn Rate          | ~25–30%                    |
| Top Churn Driver              | `customer_retention_score` |
| Peak Revenue Period           | Q4 (Oct–Dec) + Weekends    |

---

## 👤 Author

**Mandhata Pathak**

---

## 📄 License

This project is licensed under the MIT License. Feel free to use it for learning and portfolio purposes.

---

_Built with Python 🐍 | Streamlit · XGBoost · Prophet · scikit-learn _
