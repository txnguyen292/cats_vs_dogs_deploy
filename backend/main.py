"""Implement deployment logic"""
import re
import uvicorn
import requests
import sys
import base64
import tensorflow as tf
import numpy as np
import io

from typing import List, Dict, Union, Tuple
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from config import CONFIG, file_dir
from io import BytesIO

from inference import make_prediction 

model = tf.keras.models.load_model(str(CONFIG.models / "content" / "cat_vs_dog"))
app = FastAPI()
def read_image_string(contents):
    # encoded_data = contents[0].split(",")[1]
    img_decode = base64.b64decode(contents)
    np_arr = np.array(Image.open(io.BytesIO(img_decode)))
    return np_arr

@app.get("/")
def get_root():
    return {"message": "Welcome to the cat_vs_dog detection API"}

def read_imagefile(file: bytes) -> np.ndarray:
    """Load image file wit pillow

    Args:
        file (bytes): file in byte

    Returns:
        np.ndarray: numpy array representation of images
    """
    image = np.array(Image.open(BytesIO(file)))
    return image

def classify_image(model, img_arr) -> List[Dict[str, str]]:
    pred = make_prediction(model, img_arr)
    res = {"class": pred}
    resp = []
    resp.append(res)
    return resp

@app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
async def predict(contents: Dict[str, str]):
    """Make predictions on uploaded image

    Args:
        file (UploadFile, optional): file uploaded from local. Defaults to File(...).

    Returns:
        [type]: 
    """
    # extension = file.filename.split(".")[-1]
    # assert extension in ["jpg", "jpeg", "png", "jfif"], "Image must be jpg or png format"
    # img = read_imagefile(await file.read())
    # print(contents)
    img_arr = read_image_string(contents["contents"])
    predictions = classify_image(model, img_arr)
    return predictions

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)