'''import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform
import pickle

plt = platform.system()

if plt == 'Windows':
    from pathlib import PurePosixPath  # Use PurePosixPath on Windows
else:
    from pathlib import PosixPath  # Use PosixPath on Linux/Mac

import io  # Import for pre-loading image data

# Title
st.title("Fruit classification model!")

# Upload image
file = st.file_uploader("Image upload", type=['png', 'jpeg', 'img', 'jpg'])

# Load model
# Assuming you were using PosixPath for loading the model
model_path = PurePosixPath("fruits_model.pkl") if plt == 'Windows' else PosixPath("fruit_model.pkl")

# Prediction
if file is not None:
    try:
        # Pre-load image data into a buffer
        data = file.read()
        buffer = io.BytesIO(data)

        # Load model from buffer
        model = load_learner(buffer)

        # Create PIL image
        img = PILImage.create(file)

        # Display uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Perform prediction
        pred, pred_id, probs = model.predict(img)

        # Display prediction result
        st.success(pred)

        # Display prediction accuracy
        st.info(f'Accuracy: {probs[pred_id]*100:.1f}')

        # Display figure
        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)

    except (pickle.UnpicklingError, AttributeError) as e:
        # Handle potential errors during loading or prediction
        st.error(f"An error occurred: {e}")
        st.warning("Make sure the model file is not corrupted and libraries are compatible.")

'''

import streamlit as slt
from fastai.vision.all import *
import pathlib
import plotly.express as ax
import platform

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



# streamlit run .\app.py
# title
slt.title("Mevalarni aniqlaydi!")

# rasmni joylash - ya'ni tugma qo'shish
file = slt.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg', 'gif'])
if file:
    slt.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner(r"fruits_model.pkl")

    # prediction
    pred, pred_id, probs = model.predict(img)
    slt.success(f"Bashorat -> {pred}")
    slt.info(f"Ehtimollik  -> {probs[pred_id] * 100 :.1f} %")

    # plotting - ekranga ustun shaklida chiqarish
    fig = ax.bar(x=probs*100, y=model.dls.vocab)
    slt.plotly_chart(fig)
