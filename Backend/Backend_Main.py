from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from datetime import datetime

app = FastAPI()

# CORS for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if deploying securely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model pipeline
with open("best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

DATA_PATH = "CHLA_clean_data_2024_Appointments.csv"

def retrieve_appointments(clinic, start_date, end_date):
    appointments = pd.read_csv(DATA_PATH)
    appointments.columns = appointments.columns.str.strip()
    appointments['APPT_DATE'] = pd.to_datetime(appointments['APPT_DATE'])

    # Debug: Show columns in the data
    print("Columns in the dataset:", appointments.columns)

    # Ensure required columns exist
    required_columns = ['APPT_ID', 'MRN', 'CLINIC', 'APPT_DATE']
    missing_columns = [col for col in required_columns if col not in appointments.columns]

    if missing_columns:
        # Print missing columns for debugging
        print(f"Missing columns: {missing_columns}")
        return None, None

    # Filter data by clinic and date range
    filtered = appointments[
        (appointments['CLINIC'] == clinic) &
        (appointments['APPT_DATE'] >= start_date) &
        (appointments['APPT_DATE'] <= end_date)
    ]
    
    if filtered.empty:
        return None, None

    # Preserve necessary columns for prediction
    preserved_columns = filtered[['APPT_ID', 'MRN', 'CLINIC', 'APPT_DATE']].copy()

    # Debug: Check preserved columns
    print("Preserved Columns:", preserved_columns.head())

    # Drop non-predictive columns for model input
    filtered = filtered.drop(columns=['APPT_ID', 'MRN'], errors='ignore')

    return filtered, preserved_columns

def predict_no_show(appointment_data, preserved_columns):
    # Debug: Show columns before prediction
    print("Columns before prediction:", preserved_columns.columns)

    # Ensure that MRN and APPT_ID are included in the prediction data
    appointment_data_with_preserved_columns = pd.concat([appointment_data, preserved_columns[['APPT_ID', 'MRN']]], axis=1)

    # Debug: Verify columns after concatenation
    print("Columns after concatenation:", appointment_data_with_preserved_columns.columns)

    # Predict the no-show and probabilities directly using the pipeline
    predictions = pipeline.predict(appointment_data_with_preserved_columns)
    probabilities = pipeline.predict_proba(appointment_data_with_preserved_columns)[:, 1]

    # Add predictions and probabilities to the preserved columns
    preserved_columns['no_show_probability'] = probabilities
    preserved_columns['no_show_prediction'] = preserved_columns['no_show_probability'].apply(
        lambda x: 'Yes' if x >= 0.5 else 'No'
    )

    # Clean up the MRN and APPT_ID columns
    preserved_columns['MRN'] = preserved_columns['MRN'].astype(str).str.replace(',', '')
    preserved_columns['APPT_ID'] = preserved_columns['APPT_ID'].astype(str).str.replace(',', '')

    # Remove patient_id column if it exists
    if 'patient_id' in preserved_columns.columns:
        preserved_columns = preserved_columns.drop(columns=['patient_id'])

    # Reorder columns to place MRN before APPT_ID
    column_order = ['MRN', 'APPT_ID', 'CLINIC', 'APPT_DATE', 'no_show_probability', 'no_show_prediction']
    preserved_columns = preserved_columns[column_order]

    return preserved_columns

@app.get("/predict")
def get_predictions(
    clinic: str = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...)
):
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Retrieve the filtered appointments and preserved columns
        appointments, preserved = retrieve_appointments(clinic, start_dt, end_dt)

        if appointments is None:
            return {"error": "No appointments found or missing columns."}

        # Predict no-show outcomes
        results = predict_no_show(appointments, preserved)

        # Return the results in the response
        return results.to_dict(orient="records")

    except Exception as e:
        # Handle any exception and display the error message
        print(f"Error during prediction: {str(e)}")  # For debugging
        return {"error": str(e)}
