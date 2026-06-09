import streamlit as st
import requests

st.title("💊 Medication Demand Dashboard")

# Inputs utilisateur
med_id = st.number_input("Medication ID", 1)
rolling_7 = st.number_input("Rolling 7 days", 0.0)
rolling_30 = st.number_input("Rolling 30 days", 0.0)
day = st.selectbox("Day of week", list(range(7)))
month = st.selectbox("Month", list(range(1, 13)))

stock = st.number_input("Current Stock", 0.0)

if st.button("Predict & Recommend"):

    url = "http://127.0.0.1:8000/recommend"

    data = {
        "medication_id": med_id,
        "rolling_7": rolling_7,
        "rolling_30": rolling_30,
        "day_of_week": day,
        "month": month,
        "current_stock": stock
    }

    response = requests.post(url, json=data)

    result = response.json()

    st.subheader("Résultat")

    st.write(result)