# KC House Price Prediction Web App

**Project:**  
Build a Streamlit web app that showcase an EDA for the King County House prices data set, and predict 
a house price based on feautures provided by the user.  
**Files:**  
 - kc_house_data.csv: Data set file
 - App_Data_EDA.py, App_KCP.py, and App_Predict_page.py: Streamlit app files
 - Procfile, setup.sh, and requirements.txt: Files required for deployment on Heroku
 - regr.pkl and scaler.pkl: Files containing the sklearn regression and standard scaler model  
The MLP_kc folder contains the MLP model in the TF SavedModel format.

The web app is deployed on Heroku, and can be used via this [link](https://kcp-app.herokuapp.com/).
The EDA notebook with more detailed analysis is available on [Kaggle](https://www.kaggle.com/hamzaboulahia/eda-kc).
