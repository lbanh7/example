import streamlit as st
import numpy as np
import cv2
from joblib import load

st.title("TITLE")
st.header("Description of the app")

model = load("model.joblib")
st.write ("Model uploaded!")

f = st.camera_input("Take a picture")

if f is not None:
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = img[0:396, 154:550] #crop picture to (396, 396) i.e. a square
    
    # Prepare the image
    img = cv2.resize(img, (64, 64))
    img_reshaped = cv2.resize(img, (64 * 64, 1))

    #Predict the output
    y_pred = model.predict(img_reshaped)

    # Print results and plot score
    st.write(f"Predicted Letter: ", y_pred)