import streamlit as st
import numpy as np
import tensorflow as tf
from pickle import load


with open('scaler.pkl', 'rb') as file:
    scaler = load(file)

MLP_model = tf.keras.models.load_model('MLP_kc')


def show_pred_page():
    st.title("King County House Price Prediction")
    st.write("""# Provide house features to get a prediction""")

    floors=tuple(range(1, 5))
    nbr_floors = st.selectbox('Floors', options=floors)
    has_waterfront = st.radio('Waterfront', options=(True, False))

    nbr_bedrooms = st.slider(label='Bedrooms', min_value=1, max_value=35, step=1)
    nbr_bathrooms = st.slider(label='Bathrooms', min_value=0.0, max_value=10.0, step=0.25)
    liv_area = st.number_input(label='Living Area (ft2)', min_value=0, max_value=14000)
    lot_area = st.number_input(label='Lot Area (ft2)', min_value=0)
    above_area = st.number_input(label='Area Above (ft2)', min_value=0, max_value=14000)
    basement_area = st.number_input(label='Basement Area (ft2)', min_value=0)

    view = st.radio('View', options=(0, 1, 2, 3, 4))
    condition = st.radio('Condition', options=(1, 2, 3, 4, 5))
    grade = st.selectbox('Grade', options=(tuple(range(1,14))))

    built = st.slider(label='Year Built', min_value=1900, max_value=2015, step=1)

    is_renovated = st.radio('Renovated?', options=(True, False))
    if is_renovated:
        year_renov = st.slider(label='Year Renovated', min_value=1900, max_value=2015, step=1)
    else:
        year_renov = 0

    zip = st.number_input(label='Zipcode', min_value=98000, max_value=98200)

    long = st.number_input(label='Longitude', value=-122.21389640494147)
    lat = st.number_input(label='Latitude', value=47.56005251931708)

    features = [nbr_bedrooms, nbr_bathrooms, liv_area, lot_area, nbr_floors, has_waterfront, view, condition, grade,
                above_area, basement_area, built, year_renov, zip, lat, long]

    features = scaler.transform(np.array(features).reshape(-1, 16))
    st.write('  ')
    st.write('  ')
    col1, col2, col3 = st.beta_columns([1, 0.5, 1])
    if col2.button('Predict price'):
        price_pred = MLP_model.predict(np.array(features).reshape(-1, 16, 1))
        col1, col2, col3 = st.beta_columns([1, 2, 1])
        col2.subheader(f'The predicted price is: ${float(price_pred):.2f}')



