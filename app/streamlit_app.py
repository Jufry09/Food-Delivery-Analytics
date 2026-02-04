import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("linear_regression_model.pkl")

st.title("Food Delivery Time Prediction")
st.write("Predict estimated food delivery time based on order conditions.")

# Input form
weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Motorcycle", "Scooter"])

distance = st.number_input("Distance (km)", min_value=0.1, value=5.0)
prep_time = st.number_input("Preparation Time (min)", min_value=1, value=15)
experience = st.number_input("Courier Experience (years)", min_value=0, value=2)

if st.button("Predict Delivery Time"):
    input_df = pd.DataFrame({
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Time_of_Day": [time_of_day],
        "Vehicle_Type": [vehicle],
        "Distance_km": [distance],
        "Preparation_Time_min": [prep_time],
        "Courier_Experience_yrs": [experience]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")