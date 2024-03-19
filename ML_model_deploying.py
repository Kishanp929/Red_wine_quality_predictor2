import numpy as np
import pickle
import streamlit as st
import pandas as pd
import os
loaded_model = pickle.load(open('Red_wine_SVR.sav', 'rb'))

columns = [ 'fixed acidity' ,	'volatile acidity' , 	'citric acid'	, 'residual sugar'
           , 	'chlorides' ,	'free sulfur dioxide' ,	'total sulfur dioxide'	, 'density' 
               ,	'pH' ,	'sulphates' ,	'alcohol' ]
 # quality //
# input_data = [ 7.4 ,	0.700 ,	0.00 ,	1.9 ,	0.076 ,	11.0 ,
#               	34.0 ,	0.99780 ,	3.51 ,	0.56 ,	9.4	
#                ] # quality - > 5
def red_wine_quality_prediction(input_data):
    
    input_data_df = pd.DataFrame(input_data , columns = columns) 
    
    prediction = loaded_model.predict(input_data_df)
    
    # return f"Your Quality of Red Wine is {prediction}"
    return prediction


# print( red_wine_quality_prediction)
# def main():
    
    # giving a title
st.title('Red Wine Prediction Web App')
    # Add a slider widget


features_values = []
# Create sliders for each feature
fixed_acidity = st.slider("Fixed Acidity value : ", key = "1" , min_value = 4.6 , max_value=15.9, step=0.05)
features_values.append(float( fixed_acidity ))
st.write("Fixed Acidity value selected : ", fixed_acidity)

volatile_acidity = st.slider("Volatile Acidity value : ", key = "kishan" , min_value=0.12, max_value=1.58, step=0.001)
features_values.append(float(volatile_acidity))
st.write("Volatile Acidity value selected : ", volatile_acidity)

citric_acid = st.slider("Citric Acid value : ", key = "2" ,min_value=0.0, max_value=1.0, step=0.001)
features_values.append(float(citric_acid))
st.write("Citric Acid value selected : ", citric_acid)

residual_sugar = st.slider("Residual Sugar value : ", key = "3" , min_value = 0.9, max_value = 15.5, step = 0.01 )
features_values.append(float(residual_sugar))
st.write("Residual Sugar value selected : ", residual_sugar)

chlorides = st.slider("Chlorides value : ", key = "4" ,min_value=0.012,  max_value=0.611, step=0.001)
features_values.append(float(chlorides))
st.write("Chlorides value selected : ", chlorides)

free_sulfur_dioxide = st.slider("Free Sulfur Dioxide value : ", key = "5" , min_value=1.0, max_value=72.0, step=0.5)
features_values.append(float(free_sulfur_dioxide))
st.write("Free Sulfur Dioxide value selected : ", free_sulfur_dioxide)

total_sulfur_dioxide = st.slider("Total Sulfur Dioxide value : ", key = "6" , min_value=6.0, max_value=289.0, step=0.2)
features_values.append(float(total_sulfur_dioxide))
st.write("Total Sulfur Dioxide value selected : ", total_sulfur_dioxide)

density = st.slider("Density value : ", min_value=0.99, key = "10" , max_value=1.00369, step=0.000001)
features_values.append(float(density))
st.write("Density value selected : ", density)

pH = st.slider("pH value : ", min_value=2.74,key = "7" , max_value=4.01, step=0.01)
features_values.append(float(pH))
st.write("pH value selected : ", pH)

sulphates = st.slider("Sulphates value : ",key = "8" , min_value=0.33, max_value=2.00, step=0.0001)
features_values.append(float(sulphates))
st.write("Sulphates value selected : ", sulphates)

alcohol = st.slider("Alcohol value : ",key = "9" , min_value=3.0, max_value=16.0, step=0.01)
features_values.append(float(alcohol))
st.write("Alcohol value selected : ", alcohol)


    #code for Predictions 

        # Wine_quality_value = ''

    # Button For prediction

    
        
    

    
if 1:
    to_predict = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                 pH, sulphates, alcohol]
    input_data_df = pd.DataFrame( [features_values]) 
    loaded_model = pickle.load(open('C:\\Users\\Kishan\\OneDrive\\Desktop\\Model_deploying\\Red_wine_SVR.sav', 'rb'))
    # input_array  = [[ 7.4 ,	0.700 ,	0.00 ,	1.9 ,	0.076 ,	11.0 ,
    #            	34.0 ,	0.99780 ,	3.51 ,	0.56 ,	9.4	
    #              ]]
    

    prediction = float(loaded_model.predict( input_data_df) )
        
        #  Wine_quality_value = red_wine_quality_prediction(to_predict)
    # Wine_quality_value = red_wine_quality_prediction(input_array)
    # print(loaded_model.predict(input_data_df))
        
    st.write('Predictions are :'  ,  prediction)


# if __name__ == '__main__':
#     main()    

