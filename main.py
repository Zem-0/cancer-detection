import streamlit as st
import pickle
import cv2
import numpy as np

with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)
st.title('Breast Cancer Prediction')
st.write('Upload an image for prediction.')
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
    image = image.reshape(1, 50, 50, 1)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        st.write('Prediction: No Cancer')
    else:
        st.write('Prediction: Cancer')