# =============================================================
#  Retail Analytics Intelligence Platform — Streamlit Dashboard
#  Run: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Retail Analytics Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🛒 Retail Analytics")
st.sidebar.markdown("**120,000 transactions · 323 features**")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Overview",
        "📈 Sales Analysis",
        "📣 Promotion & Margin",
        "👥 Customer Segments",
        "⚠️ Churn Predictor",
        "🔮 Sales Forecast",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Dataset: [Kaggle — Supermarket]"
    "(https://www.kaggle.com/datasets/datascikhan/supermarket)"
)


# ── Data loader (cached) ──────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("supermarket_large_dataset.csv", low_memory=False)
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
    df["transaction_date"]     = pd.to_datetime(df["transaction_date"],     errors="coerce")
    df["trans_year"]      = df["transaction_datetime"].dt.year
    df["trans_month"]     = df["transaction_datetime"].dt.month
    df["trans_hour"]      = df["transaction_datetime"].dt.hour
    df["trans_weekday"]   = df["transaction_datetime"].dt.dayofweek
    df["has_discount"]    = (df["discount_percent"] > 0).astype(int)
    df["churn_label"]     = (df["customer_churn_risk"] > 0.5).astype(int)
    df["is_weekend_flag"] = df["trans_weekday"].isin([5, 6]).astype(int)
    return df


df = load_data()

# ══════════════════════════════════════════════════════════════
# PAGE ① — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛒 Retail Analytics Intelligence Platform")
    st.markdown(
        "End-to-end analytics — EDA · Segmentation · "
        "Churn Prediction · Sales Forecast · Promotion & Margin"
    )
    st.divider()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💰 Total Revenue",   f"${df['total_amount'].sum():,.0f}")
    k2.metric("🧾 Transactions",    f"{len(df):,}")
    k3.metric("👤 Customers",       f"{df['customer_id'].nunique():,}")
    k4.metric("🛒 Avg Basket",      f"${df.groupby('basket_id')['total_amount'].sum().mean():,.2f}")
    k5.metric("📊 Avg Margin",      f"{df['profit_margin_percent'].mean():.1f}%")
    k6.metric("⚠️ Churn Rate",      f"{df['churn_label'].mean()*100:.1f}%")

    st.divider()
    st.subheader("Monthly Revenue Trend")
    monthly = (
        df.assign(m=df["transaction_date"].dt.to_period("M"))
        .groupby("m")["total_amount"].sum().reset_index()
    )
    monthly["m"] = monthly["m"].astype(str)
    fig = px.bar(
        monthly, x="m", y="total_amount",
        labels={"m": "Month", "total_amount": "Revenue ($)"},
        color="total_amount", color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE ② — SALES ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Sales Analysis":
    st.title("📈 Sales Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["By Category", "By Time", "By Geography", "By Trend"]
    )

    with tab1:
        category_revenue = (
            df.groupby("product_category")["total_amount"]
            .sum().sort_values(ascending=False).head(15).reset_index()
        )
        fig = px.bar(
            category_revenue, x="total_amount", y="product_category", orientation="h",
            title="Top 15 Categories by Revenue",
            labels={"total_amount": "Revenue ($)", "product_category": "Category"},
            color="total_amount", color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        time_cols = st.columns(2)
        hourly_revenue = df.groupby("trans_hour")["total_amount"].sum().reset_index()
        fig_hour = px.bar(
            hourly_revenue, x="trans_hour", y="total_amount",
            title="Revenue by Hour", labels={"trans_hour": "Hour", "total_amount": "Revenue ($)"},
            template="plotly_white",
        )
        time_cols[0].plotly_chart(fig_hour, use_container_width=True)

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_revenue = df.groupby("trans_weekday")["total_amount"].sum().reset_index()
        weekday_revenue["day"] = weekday_revenue["trans_weekday"].map(dict(enumerate(day_names)))
        fig_weekday = px.bar(
            weekday_revenue, x="day", y="total_amount",
            title="Revenue by Weekday", labels={"day": "Weekday", "total_amount": "Revenue ($)"},
            category_orders={"day": day_names}, template="plotly_white",
        )
        time_cols[1].plotly_chart(fig_weekday, use_container_width=True)
        if "season" in df.columns:
            season_revenue = df.groupby("season")["total_amount"].sum().reset_index()
            fig_season = px.pie(
                season_revenue, names="season", values="total_amount",
                title="Revenue by Season", template="plotly_white",
            )
            st.plotly_chart(fig_season, use_container_width=True)

    with tab3:
        city_revenue = (
            df.groupby("city")["total_amount"]
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
        fig_city = px.bar(
            city_revenue, x="city", y="total_amount", title="Top 10 Cities by Revenue",
            labels={"city": "City", "total_amount": "Revenue ($)"},
            color="total_amount", color_continuous_scale="Teal", template="plotly_white",
        )
        fig_city.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

        if "region" in df.columns:
            region_revenue = df.groupby("region")["total_amount"].sum().reset_index()
            fig_region = px.pie(
                region_revenue, names="region", values="total_amount",
                title="Revenue by Region", template="plotly_white",
            )
            st.plotly_chart(fig_region, use_container_width=True)

    with tab4:
        daily_revenue = (
            df.groupby("transaction_date")["total_amount"]
            .sum().reset_index()
            .rename(columns={"transaction_date": "date", "total_amount": "revenue"})
        )
        fig_daily = px.line(
            daily_revenue, x="date", y="revenue",
            title="Daily Revenue Over Time",
            labels={"date": "Date", "revenue": "Revenue ($)"},
            template="plotly_white",
        )
        fig_daily.update_traces(line_color="#2563eb")
        st.plotly_chart(fig_daily, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE ③ — PROMOTION & MARGIN
# ══════════════════════════════════════════════════════════════
elif page == "📣 Promotion & Margin":
        st.title("📣 Promotion & Margin Monitor")
        st.caption(
            "Compare realized discount levels with sales and gross margin. "
            "These are descriptive comparisons, not causal promotion lift."
        )

        with st.container():
            category_options = sorted(df["product_category"].dropna().unique().tolist())
            selected_categories = st.multiselect("Product categories", category_options)
            promotion_filter = st.selectbox(
                "Promotion status", ["All transactions", "Promotion active", "No active promotion"]
            )

        analysis_df = df[[
            "basket_id", "product_category", "promotion_active", "discount_percent",
            "gross_sales", "gross_profit",
        ]]
        if selected_categories:
            analysis_df = analysis_df[
                analysis_df["product_category"].isin(selected_categories)
            ]
        if promotion_filter != "All transactions":
            promotion_value = "Yes" if promotion_filter == "Promotion active" else "No"
            analysis_df = analysis_df[analysis_df["promotion_active"] == promotion_value]

        if analysis_df.empty:
            st.warning("No transactions match these filters.")
        else:
            gross_sales = analysis_df["gross_sales"].sum()
            gross_profit = analysis_df["gross_profit"].sum()
            gross_margin = gross_profit / gross_sales * 100 if gross_sales else 0
            promotion_sales = analysis_df.loc[
                analysis_df["promotion_active"] == "Yes", "gross_sales"
            ].sum()
            promotion_sales_share = promotion_sales / gross_sales * 100 if gross_sales else 0

            metric_cols = st.columns(4)
            metric_cols[0].metric("Gross Sales", f"${gross_sales:,.0f}")
            metric_cols[1].metric("Gross Profit", f"${gross_profit:,.0f}")
            metric_cols[2].metric("Weighted Gross Margin", f"{gross_margin:.1f}%")
            metric_cols[3].metric("Promotion Sales Share", f"{promotion_sales_share:.1f}%")

            analysis_df = analysis_df.copy()
            analysis_df["discount_band"] = pd.cut(
                analysis_df["discount_percent"],
                bins=[-np.inf, 0, 10, 20, 30, 40, np.inf],
                labels=["No discount", "1-10%", "11-20%", "21-30%", "31-40%", "Over 40%"],
            )

            discount_summary = (
                analysis_df.groupby("discount_band", observed=False)
                .agg(
                    baskets=("basket_id", "nunique"),
                    gross_sales=("gross_sales", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    avg_discount=("discount_percent", "mean"),
                )
                .reset_index()
            )
            discount_summary["gross_margin_pct"] = np.where(
                discount_summary["gross_sales"] > 0,
                discount_summary["gross_profit"] / discount_summary["gross_sales"] * 100,
                0,
            )

            promotion_summary = (
                analysis_df.groupby("promotion_active", observed=True)
                .agg(
                    baskets=("basket_id", "nunique"),
                    gross_sales=("gross_sales", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    avg_discount=("discount_percent", "mean"),
                )
                .reset_index()
            )
            promotion_summary["gross_margin_pct"] = np.where(
                promotion_summary["gross_sales"] > 0,
                promotion_summary["gross_profit"] / promotion_summary["gross_sales"] * 100,
                0,
            )

            chart_cols = st.columns(2)
            with chart_cols[0]:
                fig = px.bar(
                    discount_summary,
                    x="discount_band", y="gross_profit", color="gross_margin_pct",
                    color_continuous_scale="RdYlGn",
                    hover_data=["baskets", "gross_sales", "avg_discount", "gross_margin_pct"],
                    labels={
                        "discount_band": "Discount band",
                        "gross_profit": "Gross profit ($)",
                        "gross_margin_pct": "Gross margin (%)",
                    },
                    title="Gross Profit by Discount Band",
                    template="plotly_white",
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with chart_cols[1]:
                fig = px.bar(
                    promotion_summary,
                    x="promotion_active", y="gross_margin_pct", color="promotion_active",
                    hover_data=["baskets", "gross_sales", "gross_profit", "avg_discount"],
                    labels={
                        "promotion_active": "Promotion active",
                        "gross_margin_pct": "Gross margin (%)",
                    },
                    title="Weighted Gross Margin: Promotion vs No Promotion",
                    template="plotly_white",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            category_summary = (
                analysis_df.groupby("product_category")
                .agg(
                    baskets=("basket_id", "nunique"),
                    gross_sales=("gross_sales", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    avg_discount=("discount_percent", "mean"),
                )
                .reset_index()
            )
            category_summary["gross_margin_pct"] = np.where(
                category_summary["gross_sales"] > 0,
                category_summary["gross_profit"] / category_summary["gross_sales"] * 100,
                0,
            )
            category_summary = category_summary.sort_values(
                ["gross_margin_pct", "avg_discount"], ascending=[True, False]
            )

            st.subheader("Category Margin Watchlist")
            st.dataframe(
                category_summary.head(15).rename(columns={
                    "product_category": "Category",
                    "baskets": "Baskets",
                    "gross_sales": "Gross sales ($)",
                    "gross_profit": "Gross profit ($)",
                    "avg_discount": "Avg discount (%)",
                    "gross_margin_pct": "Gross margin (%)",
                }).style.format({
                    "Gross sales ($)": "${:,.0f}",
                    "Gross profit ($)": "${:,.0f}",
                    "Avg discount (%)": "{:.1f}%",
                    "Gross margin (%)": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
# ══════════════════════════════════════════════════════════════
# PAGE ④ — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation — RFM + K-Means")
    st.info("RFM (Recency · Frequency · Monetary) features are computed per customer, "
            "scaled with MinMaxScaler, and clustered into 4 segments using K-Means.")

    @st.cache_data
    def compute_rfm(_df):
        snap = _df["transaction_date"].max() + pd.Timedelta(days=1)
        rfm = (
            _df.groupby("customer_id")
            .agg(
                Recency=("transaction_date",  lambda x: (snap - x.max()).days),
                Frequency=("transaction_date", "count"),
                Monetary=("total_amount",      "sum"),
            )
            .reset_index()
        )
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        rfm["Cluster"] = km.fit_predict(scaled)
        cents = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"],
        )
        names = ["Champions", "Loyal Customers", "At-Risk Customers", "Lost Customers"]
        lmap = {
            idx: names[rank]
            for rank, idx in enumerate(
                cents["Monetary"].sort_values(ascending=False).index
            )
        }
        rfm["Segment"] = rfm["Cluster"].map(lmap)
        return rfm

    rfm = compute_rfm(df)

    st.subheader("Segment Profiles")
    summary = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    summary["Customer Count"] = rfm["Segment"].value_counts()
    st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["Monetary"]),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(rfm, names="Segment", title="Customers per Segment",
                         template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        seg_m = rfm.groupby("Segment")["Monetary"].mean().reset_index()
        fig_bar = px.bar(seg_m, x="Segment", y="Monetary", color="Segment",
                         template="plotly_white",
                         labels={"Monetary": "Avg Revenue ($)"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("3-D RFM Scatter (sample 3,000 customers)")
    sample = rfm.sample(min(3000, len(rfm)), random_state=42)
    fig_3d = px.scatter_3d(
        sample, x="Recency", y="Frequency", z="Monetary",
        color="Segment", opacity=0.65,
        title="RFM Customer Segments",
        template="plotly_white",
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE ⑤ — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Predictor":
    st.title("⚠️ Customer Churn Predictor")

    MODEL_PATH   = "churn_model_xgb.pkl"
    IMPUTER_PATH = "churn_imputer.pkl"

    if not (os.path.exists(MODEL_PATH) and os.path.exists(IMPUTER_PATH)):
        st.warning(
            "⚠️ Model files not found.  \n"
            "Run **`python train_model.py`** first, then refresh this page."
        )
        st.stop()

    st.success("✅ Model loaded. Adjust sliders and click Predict.")

    with st.form("churn_form"):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)
        age      = c1.slider("Age",                          18,  80,  40)
        tenure   = c1.slider("Tenure (months)",               1, 200,  60)
        visits   = c2.slider("Avg Monthly Visits",            1,  30,   8)
        avg_txn  = c2.slider("Avg Transaction Value ($)",   5.0, 200.0, 35.0)
        clv      = c3.slider("Customer Lifetime Value ($)", 100.0, 20000.0, 3000.0)
        eng      = c3.slider("Engagement Score",            0.0,  10.0,  5.0)
        ret      = c1.slider("Retention Score",             0.0,   1.0,  0.6)
        lp_bal   = c2.slider("Loyalty Points Balance",        0, 2000, 300)
        hh_size  = c3.selectbox("Household Size",  [1, 2, 3, 4, 5], index=2)
        children = c3.selectbox("No. of Children", [0, 1, 2, 3],    index=1)
        submitted = st.form_submit_button("🔍 Predict Churn Probability")

    if submitted:
        try:
            model   = joblib.load(MODEL_PATH)
            imputer = joblib.load(IMPUTER_PATH)
            row = pd.DataFrame([{
                "customer_age":               age,
                "customer_tenure_months":     tenure,
                "average_monthly_visits":     visits,
                "average_transaction_value":  avg_txn,
                "customer_lifetime_value":    clv,
                "customer_retention_score":   ret,
                "customer_engagement_score":  eng,
                "loyalty_points_balance":     lp_bal,
                "loyalty_points_earned":      lp_bal * 1.2,
                "loyalty_points_redeemed":    lp_bal * 0.3,
                "membership_duration_months": tenure,
                "number_of_children":         children,
                "household_size":             hh_size,
                "has_discount":               1,
                "is_weekend_flag":            0,
                "customer_gender":            0,
                "customer_segment":           1,
                "customer_membership_status": 0,
                "membership_tier":            1,
                "customer_education":         1,
                "customer_occupation":        2,
            }])
            feat_names = model.get_booster().feature_names
            for c in feat_names:
                if c not in row.columns:
                    row[c] = 0
            row   = row[feat_names]
            proba = model.predict_proba(row)[0][1]

            st.divider()
            col_a, col_b = st.columns([1, 2])
            if proba >= 0.7:
                col_a.error(f"🔴 HIGH RISK\n\n**{proba*100:.1f}% churn probability**")
            elif proba >= 0.4:
                col_a.warning(f"🟡 MEDIUM RISK\n\n**{proba*100:.1f}% churn probability**")
            else:
                col_a.success(f"🟢 LOW RISK\n\n**{proba*100:.1f}% churn probability**")

            col_b.markdown("**Churn Probability**")
            col_b.progress(float(proba))

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%"},
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#ef4444" if proba > 0.5 else "#22c55e"},
                    "steps": [
                        {"range": [0,  40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE ⑥ — SALES FORECAST
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Sales Forecast":
    st.title("🔮 Sales Forecasting — Facebook Prophet")
    st.info("Prophet decomposes daily revenue into trend + weekly + yearly seasonality.")

    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

    if st.button("▶ Generate Forecast"):
        try:
            from prophet import Prophet
        except ImportError:
            st.error("Prophet not installed. Run: `pip install prophet`")
            st.stop()

        with st.spinner("Training Prophet model on daily revenue…"):
            prophet_df = (
                df.groupby("transaction_date")["total_amount"]
                .sum().reset_index()
                .rename(columns={"transaction_date": "ds", "total_amount": "y"})
                .dropna().sort_values("ds")
            )
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.1,
            )
            m.fit(prophet_df)
            future   = m.make_future_dataframe(periods=horizon, freq="D")
            forecast = m.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prophet_df["ds"], y=prophet_df["y"],
            name="Actual", line=dict(color="#3b82f6", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            name="Forecast", line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Band",
        ))
        fig.update_layout(
            title=f"Revenue Forecast — Next {horizon} Days",
            xaxis_title="Date", yaxis_title="Revenue ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Seasonality Components")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        comp_fig = m.plot_components(forecast)
        st.pyplot(comp_fig)
        plt.close("all")

        st.subheader(f"Next {horizon} Days — Forecast Table")
        future_rows = (
            forecast[forecast["ds"] > prophet_df["ds"].max()]
            [["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .rename(columns={"ds": "Date", "yhat": "Forecast ($)",
                              "yhat_lower": "Lower Bound",
                              "yhat_upper": "Upper Bound"})
            .reset_index(drop=True)
        )
        future_rows[["Forecast ($)", "Lower Bound", "Upper Bound"]] = \
            future_rows[["Forecast ($)", "Lower Bound", "Upper Bound"]].round(2)
        st.dataframe(future_rows, use_container_width=True)


