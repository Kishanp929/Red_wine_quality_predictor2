import numpy as np
import pickle
import streamlit as st
import pandas as pd
import os
loaded_model = pickle.load(open('Red_wine_SVR.sav' , 'rb'))

columns = [ 'fixed acidity' ,	'volatile acidity' , 	'citric acid'	, 'residual sugar'
           , 	'chlorides' ,	'free sulfur dioxide' ,	'total sulfur dioxide'	, 'density' 
               ,	'pH' ,	'sulphates' ,	'alcohol' ]
 # quality 
input_data = [ 7.4 ,	0.700 ,	0.00 ,	1.9 ,	0.076 ,	11.0 ,
              	34.0 ,	0.99780 ,	3.51 ,	0.56 ,	9.4	
               ] # quality - > 5

def red_wine_quality_prediction(input_data):
    
    input_data_df = pd.DataFrame( [input_data] )
    
    prediction = loaded_model.predict(input_data_df)
    
    return f"Your Quality of Red Wine is {float( prediction ) }"

def main():

    # giving a title
    st.title('Red Wine Prediction Web App')
    # Add a slider widget



# Create sliders for each feature
    fixed_acidity = st.slider("Fixed Acidity value : ", min_value = 4.0 , max_value=17.0, value=10.0, step=0.05)
    st.write("Fixed Acidity value selected : ", fixed_acidity)

    volatile_acidity = st.slider("Volatile Acidity value : ", min_value=0.08, max_value=2.3, value=1.0, step=0.01)
    st.write("Volatile Acidity value selected : ", volatile_acidity)

    citric_acid = st.slider("Citric Acid value : ", min_value=0.0, max_value=1.1, value=0.5, step=0.01)
    st.write("Citric Acid value selected : ", citric_acid)

    residual_sugar = st.slider("Residual Sugar value : ", min_value=0.7, max_value=16.0, value=2.0, step=0.01)
    st.write("Residual Sugar value selected : ", residual_sugar)

    chlorides = st.slider("Chlorides value : ", min_value=0.01, max_value=0.7, value=0.2, step=0.001)
    st.write("Chlorides value selected : ", chlorides)

    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide value : ", min_value=1.0, max_value=72.0, value=50.0, step=0.5)
    st.write("Free Sulfur Dioxide value selected : ", free_sulfur_dioxide)

    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide value : ", min_value=6.0, max_value=289.0, value=50.0, step=0.2)
    st.write("Total Sulfur Dioxide value selected : ", total_sulfur_dioxide)

    density = st.slider("Density value : ", min_value=0.8, max_value=1.2, value=1.0, step=0.005)
    st.write("Density value selected : ", density)

    pH = st.slider("pH value : ", min_value=0.0, max_value=14.0, value=10.0, step=0.01)
    st.write("pH value selected : ", pH)

    sulphates = st.slider("Sulphates value : ", min_value=0.2, max_value=2.1, value=1.0, step=0.001)
    st.write("Sulphates value selected : ", sulphates)

    alcohol = st.slider("Alcohol value : ", min_value=8.0, max_value=16.0, value=10.0, step=0.01)
    st.write("Alcohol value selected : ", alcohol)


    #code for Predictions 

    Wine_quality_value = ''

    # Button For prediction
    to_predict = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                 pH, sulphates, alcohol]
    if st.button('Check Wine Quality'):
        Wine_quality_value = red_wine_quality_prediction(to_predict)

    st.success(Wine_quality_value)


if __name__ == '__main__':
    main()    


