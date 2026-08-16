import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('car_price_model.pkl')

st.set_page_config(page_title="Pakistan Used Car Price Predictor", page_icon="🚗")

st.title("🚗 Pakistan Used Car Price Predictor")
st.write("Enter the car details below to estimate its fair market resale price (in Lacs PKR).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", [
        "Toyota", "Suzuki", "Honda", "Daihatsu", "Nissan",
        "KIA", "Hyundai", "Mitsubishi", "Mercedes", "Changan", "Other"
    ])
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2020)
    cc = st.number_input("Engine Capacity (CC)", min_value=100, max_value=6000, value=1300, step=50)
    engine_type = st.selectbox("Engine Type", ["Petrol", "Diesel", "Hybrid", "CNG", "LPG"])

with col2:
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=40000, step=1000)

st.divider()

# Exact column structure the model was trained on
EXPECTED_COLUMNS = [
    'CC', 'Km_Driven', 'Car_Age',
    'Engine_type_Diesel', 'Engine_type_Hybrid', 'Engine_type_LPG', 'Engine_type_Petrol',
    'Transmission_Manual',
    'Brand_Daihatsu', 'Brand_Honda', 'Brand_Hyundai', 'Brand_KIA', 'Brand_Mercedes',
    'Brand_Mitsubishi', 'Brand_Nissan', 'Brand_Other', 'Brand_Suzuki', 'Brand_Toyota'
]

if st.button("💰 Predict Price", use_container_width=True):

    car_age = 2026 - year

    input_dict = {
        'CC': cc,
        'Km_Driven': km_driven,
        'Car_Age': car_age,
        'Engine_type_Diesel': 1 if engine_type == "Diesel" else 0,
        'Engine_type_Hybrid': 1 if engine_type == "Hybrid" else 0,
        'Engine_type_LPG': 1 if engine_type == "LPG" else 0,
        'Engine_type_Petrol': 1 if engine_type == "Petrol" else 0,
        'Transmission_Manual': 1 if transmission == "Manual" else 0,
        'Brand_Daihatsu': 1 if brand == "Daihatsu" else 0,
        'Brand_Honda': 1 if brand == "Honda" else 0,
        'Brand_Hyundai': 1 if brand == "Hyundai" else 0,
        'Brand_KIA': 1 if brand == "KIA" else 0,
        'Brand_Mercedes': 1 if brand == "Mercedes" else 0,
        'Brand_Mitsubishi': 1 if brand == "Mitsubishi" else 0,
        'Brand_Nissan': 1 if brand == "Nissan" else 0,
        'Brand_Other': 1 if brand == "Other" else 0,
        'Brand_Suzuki': 1 if brand == "Suzuki" else 0,
        'Brand_Toyota': 1 if brand == "Toyota" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    predicted_price = model.predict(input_df)[0]

    st.divider()
    st.success(f"### Estimated Price: **{predicted_price:.2f} Lacs PKR**")
    st.caption("This is an estimate based on historical listing data and may vary from actual market conditions.")
