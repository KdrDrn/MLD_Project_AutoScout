import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:green;padding:16px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car?:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car?", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
convenience = st.sidebar.slider("How many options of your car?", 0,33, step=1) 
gears= st.sidebar.slider("How many gears of your car?", 0,8, step=1 )


de_05_model=pickle.load(open("model","rb"))
de_05_transformer = pickle.load(open('transformer', 'rb'))


my_dict = {
    'make_model': car_model,
    'Gears': gears,
    'Gearing_Type': gearing_type,
    'age': age,
    'km': km,
    'hp_kW': hp,
    'Comfort_Convenience': convenience
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)

df2 = de_05_transformer.transform(df)

st.subheader("Please press the predict button when you're ready")

if st.button("Predict"):
    prediction = de_05_model.predict(df2)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
