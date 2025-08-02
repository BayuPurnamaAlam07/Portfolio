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

tab1,tab2 = st.tabs(["Manual Input", "File Input"])


with tab1 : 
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

    #buat submit botton
    submit_botton = st.button("submit")

    if submit_botton :
        le_geography = encoder_geography.transform([geography])[0]
        le_gender = encoder_gender.transform([gender])[0]

        #merubah data input menjadi format sesuai dengan data yaitu 1 & 0
        mapping = {
            "Yes" : 1,
            "No" : 0
        }

        has_cr_card = mapping[has_cr_card]
        is_active_member = mapping[is_active_member]

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography": [le_geography],
            "Gender": [le_gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        data_scaled = scaler.transform(input_data)
        prediction = model.predict(data_scaled)
        result = prediction[0]
        prediction_proba = model.predict_proba(data_scaled)

        if result == 1:
            st.success("The customer is likely to churn")
            st.write(f"probability of churn : {prediction_proba[0][1]:.2f}")
        else :
            st.success("The customer is not likely to churn")
            st.write(f"Probability of churn: {prediction_proba[0][0]:.2f}")
            
with tab2 :
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Data from uploaded file:")
        st.dataframe(df_uploaded)
    
        df_uploaded.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True, axis=1)

        df_uploaded["Geography"] = encoder_geography.transform(df_uploaded["Geography"])
        df_uploaded["Gender"] = encoder_gender.transform(df_uploaded["Gender"])

        df_scaled = scaler.transform(df_uploaded)

        predictions = model.predict(df_scaled)
        prediction_probas = model.predict_proba(df_scaled)
        df_uploaded["Churn_Prediction"] = predictions
        df_uploaded["Churn_Probability"] = prediction_probas[:, 1]

        st.write("Predictions:")
        st.dataframe(df_uploaded[["Churn_Prediction", "Churn_Probability"]])