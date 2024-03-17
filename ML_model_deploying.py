import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('Red_wine_SVR.sav', 'rb'))

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
    
    return f"Your Quality of Red Wine is {prediction}"


def main():

    # giving a title
    st.title('Red Wine Prediction Web App')

    # Get input for 'fixed acidity'
    fixed_acidity = st.text_input('fixed acidity value:')

    # Get input for 'volatile acidity'
    volatile_acidity = st.text_input('volatile acidity value:')

    # Get input for 'citric acid'
    citric_acid = st.text_input('citric acid value:')

    # Get input for 'residual sugar'
    residual_sugar = st.text_input('residual sugar value:')

    # Get input for 'chlorides'
    chlorides = st.text_input('chlorides value:')

    # Get input for 'free sulfur dioxide'
    free_sulfur_dioxide = st.text_input('free sulfur dioxide value:')

    # Get input for 'total sulfur dioxide'
    total_sulfur_dioxide = st.text_input('total sulfur dioxide value:')

    # Get input for 'density'
    density = st.text_input('density value:')

    # Get input for 'pH'
    pH = st.text_input('pH value:')

    # Get input for 'sulphates'
    sulphates = st.text_input('sulphates value:')

    #    Get input for 'alcohol'
    alcohol = st.text_input('alcohol value:')

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

