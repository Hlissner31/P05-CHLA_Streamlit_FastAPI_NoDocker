import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.title("CHLA Clinic Appointment No-Show Prediction")

# Frontend selects clinic and date range
clinic_options = ["VALENCIA CARE CENTER", "ARCADIA CARE CENTER"]
CLINIC = st.selectbox("Select Clinic Name", clinic_options)
start_date = st.date_input("Start Date", datetime(2024, 1, 1))
end_date = st.date_input("End Date", datetime(2024, 1, 31))

if st.button("Get Predictions"):
    with st.spinner("Fetching predictions from API..."):
        # Send request to FastAPI
        try:
            response = requests.get("http://127.0.0.1:8000/predict",  # Use backend service URL in Docker network
                params={
                    "clinic": CLINIC,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                df = pd.DataFrame(result)
                st.subheader("No-Show Predictions")
                st.dataframe(df.style.hide(axis="index"))

                st.download_button(
                    label="Download Predictions",
                    data=df.to_csv(index=False),
                    file_name="no_show_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
