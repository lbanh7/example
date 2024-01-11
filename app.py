import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer
import av


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


st.title("TITLE")
st.header("Description of the app")

ALPHABET = ['A', 'B', 'C', 'D', 'E',
              'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z']

model = tf.keras.models.load_model("best_cnn_model.h5")
st.write ("Model uploaded!")

# f = st.camera_input("Take a picture")
# f = camera_input_live()
f = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

height, width = 48, 48

if f is not None:
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = image[0:396, 154:550]

    # Prepare the image
    resized = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)
    gray = np.mean(resized, axis=-1)

    normalized = gray/255
    model_input = np.expand_dims(normalized,0)

    #Predict the output
    scores = model.predict(model_input).flatten()

    # Print results
    st.write(f"Translation: {ALPHABET[scores.argmax()]}")

    f = st.camera_input("Take a picture")