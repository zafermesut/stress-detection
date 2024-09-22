import streamlit as st
import neattext.functions as nfx
import joblib

def clean_text(text):
    text = nfx.remove_special_characters(text)  
    text = nfx.remove_stopwords(text)  
    text = nfx.remove_numbers(text)  
    text = nfx.normalize(text)
    return text

model_pipeline = joblib.load('model.joblib')

st.title("Stress Detection App")

st.write("This is a simple app to detect stress in text data")

user_input = st.text_area("Text", "")

if st.button("Predict"):
    if user_input:
        prediction = model_pipeline.predict([user_input])
        result = prediction[0]
        st.success(f"Result: {result}")
    else:
        st.warning("Please enter a text to predict")
