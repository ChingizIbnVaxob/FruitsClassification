import streamlit as slt
from fastai.vision.all import *
import pathlib
import plotly.express as ax
import platform

import pathlib

print(pathlib.__version__)


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



# streamlit run .\app.py
# title
slt.title("Mevalarni aniqlaydi!")

# image uploader
file = slt.file_uploader("Image Upload", type=['png', 'jpg', 'jpeg', 'gif'])
if file:
    slt.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner("fruits_model.pkl")

    # prediction
    pred, pred_id, probs = model.predict(img)
    slt.success(f"Prediction -> {pred}")
    slt.info(f"Accuracy  -> {probs[pred_id] * 100 :.1f} %")

    # plotting
    fig = ax.bar(x=probs*100, y=model.dls.vocab)
    slt.plotly_chart(fig)
