import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- STEP 1: WEBSITE SETUP ---
st.set_page_config(page_title="House ML Master", layout="wide")
# --- STEP 1: WEBSITE SETUP ---
st.set_page_config(page_title="House ML Master", layout="wide")

# Updated CSS to work perfectly with both Dark and Light modes
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1); /* Semi-transparent grey */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
st.title("🏠 House Price ML: Analysis to Prediction")
st.write("This application demonstrates the complete Machine Learning workflow, from data exploration to real-time prediction.")

# --- STEP 2: LOAD & TRAIN (Backend Logic) ---
@st.cache_data 
def load_and_train():
    # Fetching dataset from Scikit-Learn
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Using Gradient Boosting as per Task 6 instructions
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation Metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    return df, model, scaler, metrics

df, model, scaler, metrics = load_and_train()

# --- STEP 3: DISPLAY ANALYSIS (EDA) ---
st.divider()
st.header("📊 Part 1: Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.write("### Dataset Preview")
    st.write("Displaying the top 10 records of the California Housing dataset.")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.write("### Feature Correlation Heatmap")
    st.write("Visualizing how different features correlate with the target price.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- STEP 4: DISPLAY PERFORMANCE ---
st.divider()
st.header("🏆 Part 2: Model Performance Metrics")
m1, m2 = st.columns(2)
# Displaying metrics in a professional dashboard style
m1.metric("Model Accuracy (R2 Score)", f"{metrics['R2']:.4f}")
m2.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.4f}")

# --- STEP 5: PREDICTION UI ---
st.divider()
st.header("🔮 Part 3: Predict House Price")
st.write("Provide the property details below to generate an estimated market value.")

# Input fields organized in columns
c1, c2, c3 = st.columns(3)
med_inc = c1.number_input('Median Income (in $10k)', value=3.5)
house_age = c1.number_input('House Age (Years)', value=28.0)
ave_rooms = c2.number_input('Average Rooms', value=5.0)
ave_bedrms = c2.number_input('Average Bedrooms', value=1.0)
population = c3.number_input('Local Population', value=1400.0)
ave_occup = c3.number_input('Average Occupancy', value=3.0)

# Latitude and Longitude Sliders
lat = st.slider('Latitude', 32.0, 42.0, 35.0)
long = st.slider('Longitude', -124.0, -114.0, -120.0)

# Prediction Button Logic
if st.button("Generate Price Estimate"):
    # Formatting input for the model
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, long]], 
                              columns=df.columns[:-1])
    
    # Transforming input using the trained scaler
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    # Displaying the final result (Scaled back to original price)
    st.success(f"### 💰 Estimated Market Price: ${prediction[0]*100000:,.2f}")
    st.balloons()

# --- FOOTER ---
st.divider()
st.caption("Developed for DevelopersHub AI/ML Internship | Task 6: House Price Prediction")