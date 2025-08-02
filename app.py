import streamlit as st
import pandas as pd
import joblib

#baca data
df = pd.read_csv("Churn_Modelling.csv")
#ambil uniq value dari kolom kategorical
geolist = df["Geography"].unique().tolist()
genderlist = df["Gender"].unique().tolist()

#load model
model = joblib.load("model.pkl")
encoder_geography = joblib.load("Geography-encoder.pkl")
encoder_gender = joblib.load("Gender-encoder.pkl")
scaler = joblib.load("scaler (1).pkl")

#membuat title 
st.title("Churn Prediction App")

#bikin lay out streamlit menjadi beberapa kolom
col1, col2, col3 = st.columns(3)
with col1:
    #input untuk credit score ( number input ) 
    # min_value dan max_value sesuai dengan data yang dipakai untuk training 
    credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=600)
    
    #input untuk geography ( selectbox )
    # options nya diambil dari dataframe
    geography = st.selectbox("Geography", options=geolist, index=0)
    
    #input untuk gender ( selectbox )
    # options nya diambil dari dataframe
    gender = st.selectbox("Gender", options=genderlist, index=0)

with col2:
    
    # age ( slider )
    # min_value dan max_value sesuai dengan data yang dipakai untuk training
    age = st.slider("Age", min_value=18, max_value=100, value=30)

    #tenure ( slider )
    # min_value dan max_value sesuai dengan data yang dipakai untuk training
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

    #input untuk balance ( number input )
    # min_value dan max_value sesuai dengan data yang dipakai untuk training
    balance = st.number_input("Balance", min_value=0, max_value=250000, value=5000)

with col3:
    #input untuk number of products ( selectbox )
    # options nya adalah [1, 2, 3, 4] -> sesuai dengan data
    num_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=1)

    #input untuk has credit card ( selectbox )
    # options nya adalah ["Yes", "No"] -> di dataset nya 1 dan 0 jadi perlu untuk di mapping lagi nanti
    has_cr_card = st.selectbox("Has Credit Card", options=["Yes", "No"], index=0)

    #is active member ( selectbox )
    # options nya adalah ["Yes", "No"] -> di dataset nya 1 dan 0 jadi perlu untuk di mapping lagi nanti
    is_active_member = st.selectbox("Is Active Member", options=["Yes", "No"], index=0)

    #input untuk estimated salary ( number input )
    # min_value dan max_value sesuai dengan data yang dipakai untuk training
    estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=200000, value=50000)
    