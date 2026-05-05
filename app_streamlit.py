import streamlit as st
import joblib
import numpy as np
import pandas as pd
 
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
 
st.title("🚗 Car Price Predictor")
st.markdown("Predict the price of a used car based on its features.")
 
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    unique_values = joblib.load("unique_values.pkl")
    return model, encoder, scaler, unique_values
 
model, encoder, scaler, unique_values = load_artifacts()
 
st.subheader("Enter Car Details")
 
col1, col2 = st.columns(2)
 
with col1:
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2015)
    odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=80000, step=1000)
    manufacturer = st.selectbox("Manufacturer", sorted([v for v in unique_values['manufacturer'] if isinstance(v, str)]))
    condition = st.selectbox("Condition", sorted([v for v in unique_values['condition'] if isinstance(v, str)]))
 
with col2:
    fuel = st.selectbox("Fuel", sorted([v for v in unique_values['fuel'] if isinstance(v, str)]))
    transmission = st.selectbox("Transmission", sorted([v for v in unique_values['transmission'] if isinstance(v, str)]))
    drive = st.selectbox("Drive", sorted([v for v in unique_values['drive'] if isinstance(v, str)]))
    vehicle_type = st.selectbox("Type", sorted([v for v in unique_values['type'] if isinstance(v, str)]))
 
state = st.selectbox("State", sorted([v for v in unique_values['state'] if isinstance(v, str)]))
 
if st.button("💰 Predict Price", use_container_width=True):
    num_data = np.array([[float(year), float(odometer)]])
    num_scaled = scaler.transform(num_data)
 
    cat_data = np.array([[manufacturer, condition, fuel, transmission, drive, vehicle_type, state]])
    cat_encoded = encoder.transform(cat_data)
 
    X_final = np.hstack([num_scaled, cat_encoded])
    prediction = model.predict(X_final)[0]
 
    st.success(f"### Estimated Price: **${prediction:,.0f}**")
    st.caption("Prediction based on XGBoost model trained on Craigslist used cars dataset.")