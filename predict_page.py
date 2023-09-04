import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("housedata.pkl", "rb") as file:
        webdata = pickle.load(file)

    return webdata

webdata = load_model()

model_loaded = webdata["model"]
scaler_loaded = webdata["scaler"]

def show_predict_page():
    st.title("House Price Prediciton")

    st.write("""### We need some information to predict the salary""")

    qual = st.slider('Overall Quality of House', 1, 10)
    st.write("Your house Quality is", qual, '.')
    #
    exterQual = st.slider(
        'Quality of the exterior material of the property.',1, 5)
    st.write("Your house Quality is", exterQual, '.')
    
    bsmtQual = st.slider('Overall Basement Quality of House', 1, 5)
    st.write("Your house Quality is", bsmtQual, '.')
    #
    totalBsmtSF = st.number_input('Total Basement Square Footage')
    st.write('The current number is ', totalBsmtSF)
    st.write("Note: Max number you can put 3200.")
    #
    stFlrSF = st.slider('First Floor Square Meter of House', 5.8, 9.0)
    st.write("Your house Quality is", stFlrSF, '.')
    #
    grLivArea = st.slider('Above Ground Living Area(meter) of House', 5.8, 8.5)
    st.write("Your house Quality is", grLivArea, '.')
    
    
    fullBath = st.selectbox(
        'How many full bathrooms in a property.',
        ('0', '1', '2', "3"))
    st.write('You selected:', fullBath)
    #
    KitchenQual = st.slider(
        'The quality of the kitchen in the property.',1, 5)
    st.write("Your house Quality is", KitchenQual, '.')
    

    garageCars = st.slider('The capacity of the garage in terms of the number of cars', 0, 4)
    st.write("Your house Quality is", garageCars, '.')
    #
    garageArea = st.number_input('Total square footage of the garage in a property')
    st.write('The current number is ', garageArea)
    st.write("Note: Max number you can put 1390.")

    ok = st.button("Calculate Price")
    if ok:
        a = np.array([qual, exterQual, bsmtQual, totalBsmtSF, stFlrSF, grLivArea, fullBath, KitchenQual, garageCars, garageArea])
        a = a.reshape(1,-1) 
        a = scaler_loaded.transform(a)
        

        price = model_loaded.predict(a)
        st.subheader(f"The estimated price is ${price[0][0]}")