import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('quikr_car.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

company = st.selectbox('Select Car Brand', cars_data['company'].unique())
year = st.slider('Car Manufactured Year', 1994,2024)
kms_driven = st.slider('No of kms Driven', 11,200000)
fuel_type = st.selectbox('Fuel type', cars_data['fuel_type'].unique())


if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[company,year,kms_driven,fuel_type]],
    columns=['company','year','kms_driven','fuel_type'])
    
    input_data_model['fuel_type'].replace(['Diesel', 'Petrol', 'LPG'],[1,2,3], inplace=True)

    input_data_model['company'].replace(['Hyundai', 'Mahindra', 'Maruti', 'Ford', 'Skoda', 'Audi', 'Toyota',
       'Renault', 'Honda', 'Datsun', 'Mitsubishi', 'Tata', 'Volkswagen',
       'Chevrolet', 'Mini', 'BMW', 'Nissan', 'Hindustan', 'Fiat', 'Force',
       'Mercedes', 'Land', 'Jaguar', 'Jeep', 'Volvo'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
                          ,inplace=True)

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be '+ str(car_price[0]))