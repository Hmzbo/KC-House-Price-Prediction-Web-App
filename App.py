import streamlit as st
from Predict_page import show_pred_page
from Data_EDA import show_EDA_page

page = st.sidebar.selectbox('Page', options=('Data EDA','Predict'))

if page == 'Predict':
    show_pred_page()
else:
    show_EDA_page()