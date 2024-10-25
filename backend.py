import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st


save_dir = r'D:\path_to_save'
model_path = os.path.join(save_dir, 'model.pkl')
scaler_path = os.path.join(save_dir, 'scaler.pkl')


with open(model_path, 'rb') as model_file:
    diabetes_model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


image_model_path = 'D:/path_to_your_model/your_model.keras'
image_model = tf.keras.models.load_model(image_model_path)

def main():
    st.title("Diabetes Prediction App")
    st.sidebar.header("User Input")

    
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, step=1)   
    glucose = st.sidebar.number_input('Glucose Level', 0, 200)
    blood_pressure = st.sidebar.number_input('Blood Pressure', 0, 122)
    skin_thickness = st.sidebar.number_input('Skin Thickness', 0, 99)
    insulin = st.sidebar.number_input('Insulin', 0, 846)
    bmi = st.sidebar.number_input('BMI', 0.0, 67.1)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 2.5)
    age = st.sidebar.number_input('Age', 21, 100)

    st.sidebar.subheader("Upload a Retinal Image (optional):")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.sidebar.button("Predict"):
        
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_data_scaled = scaler.transform(input_data)

        
        diabetes_prediction = diabetes_model.predict(input_data_scaled)[0]
        st.write(f"Diabetes Prediction (Logistic Regression): {diabetes_prediction}")

        
        if uploaded_file is not None:
            img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(200, 200))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

           
            image_prediction = image_model.predict(img_array)
            image_prediction = 1 if image_prediction[0][0] > 0.5 else 0
            st.write(f"Image Prediction (CNN): {image_prediction}")
        else:
            image_prediction = 0  

        
        if diabetes_prediction == 0 and image_prediction == 1:
            result = "Diabetes risk is less than 10%."
        elif diabetes_prediction == 1 and image_prediction == 1:
            result = "Diabetes risk is below 50%."
        elif diabetes_prediction == 1 and image_prediction == 0:
            result = "Diabetes risk is above 50%."
        else:
            result = "Risk assessment could not be determined."

        
        st.success(result)

if __name__ == '__main__':
    main()
