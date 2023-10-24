import streamlit as st
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import PIL
from PIL import Image
import imageio as iio

model_path = "potatoes.h5"

model = load_model(model_path)
st.write("Upload Leaf image to identify disease or healty!!")
img_uploaded = st.file_uploader("Choose a file",type="jpg")

if img_uploaded is not None:
 img_width, img_height = 256, 256
 img = iio.imread(img_uploaded)
 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 st.image(img)
 img = Image.open(img_uploaded)
 img = img.resize((img_width, img_height))
 img_array = np.array(img)
 input_data = np.expand_dims(img_array, axis=0)
 input_data = input_data / 255.0  # Normalize the input if necessary
 output = model.predict(input_data)
 print(output)
 classes = ['Early_blight','Late_Blight','Healthy']
 st.write(classes[output.argmax()])

