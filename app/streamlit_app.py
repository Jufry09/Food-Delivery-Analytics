import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Food Delivery Dashboard", layout="wide")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("linear_regression_model.pkl")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed/food_delivery_clean.csv")
    return df

df = load_data()

# -------------------------
# Sidebar for user input
# -------------------------
st.sidebar.header("Input Delivery Features")

weather = st.sidebar.selectbox("Weather", df['Weather'].unique())
traffic = st.sidebar.selectbox("Traffic Level", df['Traffic_Level'].unique())
time_of_day = st.sidebar.selectbox("Time of Day", df['Time_of_Day'].unique())
vehicle = st.sidebar.selectbox("Vehicle Type", df['Vehicle_Type'].unique())
distance = st.sidebar.number_input("Distance (km)", min_value=0.1, value=5.0)
prep_time = st.sidebar.number_input("Preparation Time (min)", min_value=1, value=15)
experience = st.sidebar.number_input("Courier Experience (years)", min_value=0, value=2)

input_df = pd.DataFrame({
    "Weather": [weather],
    "Traffic_Level": [traffic],
    "Time_of_Day": [time_of_day],
    "Vehicle_Type": [vehicle],
    "Distance_km": [distance],
    "Preparation_Time_min": [prep_time],
    "Courier_Experience_yrs": [experience]
})

# -------------------------
# Predict button
# -------------------------
if st.sidebar.button("Predict Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.sidebar.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

# -------------------------
# Main Page - Dashboard & EDA
# -------------------------
st.title("Food Delivery Analytics Dashboard")

# Dataset overview
st.markdown("### Dataset Overview")
st.dataframe(df.head(10))

# Distribution of Delivery Time
st.markdown("### Delivery Time Distribution")
plt.figure(figsize=(8,4))
sns.histplot(df["Delivery_Time_min"], bins=30, kde=True)
plt.xlabel("Delivery Time (minutes)")
st.pyplot(plt.gcf())
plt.clf()

# Boxplot by Traffic Level
st.markdown("### Delivery Time by Traffic Level")
plt.figure(figsize=(8,4))
sns.boxplot(x="Traffic_Level", y="Delivery_Time_min", data=df)
st.pyplot(plt.gcf())
plt.clf()

# Boxplot by Weather
st.markdown("### Delivery Time by Weather")
plt.figure(figsize=(8,4))
sns.boxplot(x="Weather", y="Delivery_Time_min", data=df)
st.pyplot(plt.gcf())
plt.clf()

# Scatter plots numeric vs Delivery Time
st.markdown("### Numeric Features vs Delivery Time")
numeric_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=col, y="Delivery_Time_min", data=df)
    plt.xlabel(col)
    plt.ylabel("Delivery Time")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------
# Model Performance Metrics
# -------------------------
st.markdown("### Model Performance on Test Set")

X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"**MAE:** {mae:.2f} minutes")
st.write(f"**RMSE:** {rmse:.2f} minutes")
st.write(f"**R²:** {r2:.3f}")

# Actual vs Predicted scatter plot
st.markdown("### Actual vs Predicted Delivery Time")
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")
st.pyplot(plt.gcf())
plt.clf()

st.markdown("### End of Dashboard")