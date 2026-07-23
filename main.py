# Import essential libraries

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load model
model=tf.keras.models.load_model("ann_model.keras")

# Title
st.title("Digit Recognition System Using ANN")

# Upload flie
uploaded_file=st.file_uploader("Choose image",type=["jpg",'jpeg',"png"])

if uploaded_file is not None:
    # Open Image
    image=Image.open(uploaded_file).convert('L')
    # L = Luminance
    # Image will Contain only shades of grey
    st.image(image, caption="Uploaded Image", width=150)
    #Copnvert image to array
    img=np.array(image)
    # Resize to 28x28
    img=cv2.resize(img, (28,28))
    # Invert img color
    img=255-img
    #normalize
    img=img/255.0
    # Reshape for prediction
    img=img.reshape(1,28,28)
    # prediction
    prediction=model.predict(img)
    predicted_digit=np.argmax(prediction)
    st.success(f"Expected Digit = {predicted_digit}")