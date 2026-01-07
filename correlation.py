import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def show_environment_correlation(
    df: pd.DataFrame,
    selected_disease: str,
    bins: int = 10
):
    """
    Displays environmental correlation plots for a given disease.

    Expected columns in df:
    - disease
    - disease_percentage
    - temperature_c
    - humidity_pct
    """

    if selected_disease not in df["disease"].unique():
        st.warning("Environmental correlation not available for this disease.")
        return

    disease_df = df[df["disease"] == selected_disease]

    st.subheader("🌡 Environmental Correlation Analysis")

    col1, col2 = st.columns(2)

    # ---------- Humidity (Binned Line) ----------
    with col1:
        disease_df["humidity_bin"] = pd.cut(
            disease_df["humidity_pct"], bins=bins
        )

        hum_agg = (
            disease_df
            .groupby("humidity_bin", observed=True)
            ["disease_percentage"]
            .mean()
            .reset_index()
        )

        hum_x = hum_agg["humidity_bin"].apply(lambda x: x.mid)

        fig1, ax1 = plt.subplots()
        ax1.plot(hum_x, hum_agg["disease_percentage"], marker="o")
        ax1.set_xlabel("Humidity (%)")
        ax1.set_ylabel("Avg Disease Severity (%)")
        ax1.set_title("Avg Disease Severity vs Humidity")
        st.pyplot(fig1)

    # ---------- Temperature (Binned Line) ----------
    with col2:
        disease_df["temp_bin"] = pd.cut(
            disease_df["temperature_c"], bins=bins
        )

        temp_agg = (
            disease_df
            .groupby("temp_bin", observed=True)
            ["disease_percentage"]
            .mean()
            .reset_index()
        )

        temp_x = temp_agg["temp_bin"].apply(lambda x: x.mid)

        fig2, ax2 = plt.subplots()
        ax2.plot(temp_x, temp_agg["disease_percentage"], marker="o")
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Avg Disease Severity (%)")
        ax2.set_title("Avg Disease Severity vs Temperature")
        st.pyplot(fig2)