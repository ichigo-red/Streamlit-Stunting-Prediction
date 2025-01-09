import streamlit as st
import pandas as pd
import joblib 
import numpy as np

try:
    from custom_rf import HistoryRandomForest  
except ImportError:
    class HistoryRandomForest:
        pass 

model = joblib.load(open('history_random_forest.pkl', 'rb')) 
data = pd.read_csv('Stunting_Cleaningdata.csv')


st.title("Prediksi Stunting pada Balita")
st.write("Aplikasi ini menggunakan model Random Forest untuk memprediksi kemungkinan stunting.")

gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
gender_int = 0 if gender == "Laki-laki" else 1
age = st.number_input("Umur (bulan)", min_value=0, max_value=50, step=1)
birth_weight = st.number_input("Berat Lahir (kg)", min_value=0., max_value=10.0, step=0.1)
birth_length = st.number_input("Panjang Lahir (cm)", min_value=0, step=1)
body_weight = st.number_input("Berat Badan Sekarang (kg)", min_value=0., max_value=25.0,step=0.1)
body_length = st.number_input("Panjang Badan Sekarang (cm)", min_value=0., max_value=100.0, step=0.1)

input_data = np.array([[gender_int, age, 
                        birth_weight, birth_length, 
                        body_weight, body_length]])

def predict(input_data):
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    return model.predict(input_data)

if st.button("Prediksi"):
    result = predict(input_data)
    if result[0] == 0:
        st.success("Hasil: Anak tidak mengalami stunting.")
    else:
        st.error("Hasil: Anak mengalami stunting.")
