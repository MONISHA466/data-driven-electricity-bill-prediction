import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from recommendation import appliance_contributions, normalize_importances


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "bill_model.joblib"
DATA_PATH = BASE_DIR / "data_sample.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python model_training.py' first to train and save the model."
        )
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    return model, feature_cols


@st.cache_data
def load_historical_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
    df = df.sort_values("Month")
    return df


def render_visualizations(insights=None, feature_values=None, key_prefix="viz"):
    """Render energy consumption charts. Use key_prefix to avoid duplicate element IDs."""
    # --- Historical monthly trend (from sample data) ---
    hist_df = load_historical_data()
    if hist_df is not None:
        st.subheader("📊 Monthly Energy Consumption Trend")
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=hist_df["Month"],
                y=hist_df["Units_Consumed"],
                name="Units (kWh)",
                marker_color="#00C853",
                yaxis="y",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["Month"],
                y=hist_df["Total_Bill"],
                name="Bill (₹)",
                line=dict(color="#FF6D00", width=3),
                yaxis="y2",
            )
        )
        fig_trend.update_layout(
            title="Monthly Units Consumed vs Bill",
            xaxis_title="Month",
            yaxis=dict(title=dict(text="Units (kWh)", font=dict(color="#00C853"))),
            yaxis2=dict(
                title=dict(text="Bill (₹)", font=dict(color="#FF6D00")),
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_trend, use_container_width=True, key=f"{key_prefix}_trend")

        # --- Appliance usage by month (stacked bar) ---
        st.subheader("📈 Appliance Usage by Month (hours)")
        appliance_cols = ["Washing_Machine_Usage", "AC_Usage", "Heater_Usage"]
        fig_app = go.Figure()
        colors = {"Washing_Machine_Usage": "#2196F3", "AC_Usage": "#FF9800", "Heater_Usage": "#9C27B0"}
        for col in appliance_cols:
            fig_app.add_trace(
                go.Bar(
                    x=hist_df["Month"],
                    y=hist_df[col],
                    name=col.replace("_", " ").replace("Usage", ""),
                    marker_color=colors.get(col, "#607D8B"),
                )
            )
        fig_app.update_layout(
            barmode="stack",
            title="Cumulative Appliance Usage per Month",
            xaxis_title="Month",
            yaxis_title="Total Hours",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_app, use_container_width=True, key=f"{key_prefix}_appliance")

    # --- Prediction-based charts (appliance contribution) ---
    if insights and feature_values:
        st.subheader("🔌 Your Usage: Appliance Impact on Bill")
        names = [i.name.replace("_", " ") for i in insights]
        contribs = [i.relative_importance * 100 for i in insights]
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=names,
                    values=contribs,
                    hole=0.45,
                    marker_colors=["#00C853", "#2196F3", "#FF9800"],
                )
            ]
        )
        fig_pie.update_layout(
            title="Relative Impact on Predicted Bill",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            template="plotly_white",
            height=380,
            annotations=[dict(text="Appliance<br>Impact", x=0.5, y=0.5, font_size=14, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"{key_prefix}_pie")

        # Bar chart of contribution
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=names,
                    y=contribs,
                    marker_color=["#00C853", "#2196F3", "#FF9800"],
                    text=[f"{v:.1f}%" for v in contribs],
                    textposition="outside",
                )
            ]
        )
        fig_bar.update_layout(
            title="Appliance Contribution to Bill (%)",
            xaxis_title="Appliance",
            yaxis_title="Impact (%)",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")


def main():
    st.set_page_config(page_title="Electricity Bill Prediction", layout="wide")
    st.title("⚡ Smart Electricity Bill Predictor")
    st.markdown(
        "Advanced system that predicts your **electricity bill**, "
        "highlights **energy‑hungry appliances**, and suggests **how to reduce consumption**."
    )

    # Tabs: Visualizations | Prediction
    tab_viz, tab_pred = st.tabs(["📊 Energy Consumption Visualizations", "🔮 Bill Prediction"])

    with tab_viz:
        render_visualizations(key_prefix="tab_viz")

    model, feature_cols = load_model()

    with tab_pred:
        st.subheader("1️⃣ Enter your usage details")

        col1, col2 = st.columns(2)

        with col1:
            units_consumed = st.number_input(
                "Total units consumed this month (kWh)",
                min_value=0.0,
                value=300.0,
                step=10.0,
            )
            washing_hours = st.number_input(
                "Washing machine usage (hours/day)",
                min_value=0.0,
                value=1.0,
                step=0.5,
            )
            ac_hours = st.number_input(
                "AC usage (hours/day)",
                min_value=0.0,
                value=4.0,
                step=0.5,
            )
        with col2:
            heater_hours = st.number_input(
                "Heater usage (hours/day)",
                min_value=0.0,
                value=0.5,
                step=0.5,
            )
            temperature = st.number_input(
                "Average outdoor temperature (°C)",
                min_value=-5.0,
                value=28.0,
                step=1.0,
            )

        if st.button("Predict Bill"):
            feature_values = {
                "Units_Consumed": units_consumed,
                "Washing_Machine_Usage": washing_hours,
                "AC_Usage": ac_hours,
                "Heater_Usage": heater_hours,
                "Temperature": temperature,
            }

            X = np.array([[feature_values[c] for c in feature_cols]], dtype=float)
            predicted_bill = float(model.predict(X)[0])

            st.subheader("2️⃣ Predicted Bill")
            st.metric(label="Estimated Monthly Bill", value=f"₹{predicted_bill:,.2f}")

            # Appliance‑wise contribution using model feature importances
            importances_raw = dict(zip(feature_cols, model.feature_importances_))
            importances_norm = normalize_importances(importances_raw)
            insights = appliance_contributions(importances_norm, feature_values)

            st.subheader("3️⃣ Appliance Consumption Analysis")
            st.markdown("Appliances are ordered by their **relative impact** on your bill.")

            for insight in insights:
                st.markdown(
                    f"**{insight.name.replace('_', ' ')}**  \n"
                    f"- Usage: **{insight.usage_hours:.1f} hours/day**  \n"
                    f"- Estimated contribution: **{insight.relative_importance * 100:.1f}%**  \n"
                    f"- Suggestion: {insight.suggestion}"
                )

            st.subheader("4️⃣ Consumption Visualizations")
            render_visualizations(insights=insights, feature_values=feature_values, key_prefix="tab_pred")

            st.subheader("5️⃣ Summary Recommendations")
            st.markdown(
                "- **Reduce AC and heater usage during peak hours when possible.**  \n"
                "- **Run washing machine with full loads and fewer cycles per week.**  \n"
                "- **Turn off idle appliances and improve insulation to reduce heating/cooling loads.**"
            )


if __name__ == "__main__":
    main()

