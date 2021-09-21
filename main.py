import streamlit as st
import keras
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import urllib.request
import time

html_temp = '''
    <div style = "background-color: rgba(25,25,112,0.03); padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><font size="6" face="verdana" color="green"><h1>Handwritten Digit Recognition</h1></font></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload Image for Classification</h3></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select','Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)
try:
    if image is not None:
        st.image(image,width=300,caption="Uploaded Image")
        if st.button('Predict'):
            model=keras.models.load_model('Model/model.h5')
            image = image.convert('RGB')
            image = np.array(image)
            print(image) 
            # Convert RGB to BGR 
            image = image[:, :, ::-1].copy() 
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
            nor=tf.keras.utils.normalize(resized,axis=1)
            nor=np.array(nor).reshape(-1,28,28,1)
            m=model.predict(nor)
            st.success('The uploaded digit is {}'.format(np.argmax(m)))
            st.balloons()
except:
    st.info("Please upload image in .jpg , .jpeg or .png format")
    pass
