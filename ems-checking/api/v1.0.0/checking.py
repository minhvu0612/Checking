from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from training.src.make_data_train import Trainer
from training.src.face_rec import Face_recognition
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import base64
import io
from training.src.settings import (
    DATA_FACE_DIR,
)


# Class 
class Member(BaseModel):
    email: str
    images: str

class Detection(BaseModel):
    image: str

class Elimination(BaseModel):
    email: str

class Login(BaseModel):
    email: str
    password: str

class UnicornException(Exception):
    def __init__(self, email: str):
        self.email = email

# Router
@app.post("/tims/api/v1/checking")
async def add_new_member(item: Member):
    trainer = Trainer()
    indexes = []
    s = ''
    df = pd.read_json(DATA_FACE_DIR)
    members = df['email'].to_numpy()
    if item.email in members:
        return "Email already exist!"
    if len(item.images) < 5:
        return "Not enough pictures!"
    for index in range(len(item.images)):
        img = stringToRGB(item.images[index])
        result, index = trainer.add_member(image = img, email = item.email, index=index)
        if result is None:
            indexes.append(index)
    if len(indexes) == 0:
        return "Add member successfully!"
    else:
        for index in indexes[:-1]:
            s = s + ' ' + str(index) + ','
        s = s + ' ' + str(index) + '!'
        return "Add member failed! Images that cannot be recognized:" + s 

@app.delete("/delete")
async def delete_member(item: Elimination):
    return item.email

@app.post("/detection")
async def detection_face(item: Detection):
    #image = base64.b64encode(item.image)
    image = stringToRGB(item.image)
    result = face_recogny.recogny_face(image)
    frame, bbox, email, current_time = result
    frame_b64 = RGB2String(result[0])
    return  email, current_time

@app.get("/tims/api/v1/checking")
async def hello():
    response = ExceptionType(200, "CC").getResponse()
    return response