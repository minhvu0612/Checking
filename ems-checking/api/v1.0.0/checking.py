from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from training.src.make_data_train import Trainer
from training.src.face_rec import Face_recognition
import numpy as np
from PIL import Image
import cv2
import base64
import io



# Class 
class Member(BaseModel):
    name: str
    image: str

class Detection(BaseModel):
    image: str

class Elimination(BaseModel):
    name: str

class Login(BaseModel):
    email: str
    password: str

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

# Router
@app.post("/tims/api/v1/checking")
async def add_new_member(item: Member):
    trainer = Trainer()
    img = stringToRGB(item.image)
    trainer.add_member(image = img, name = item.name)
    return "Add member successfully!"

@app.delete("/delete")
async def delete_member(item: Elimination):
    return item.name

@app.post("/detection")
async def detection_face(item: Detection):
    #image = base64.b64encode(item.image)
    image = stringToRGB(item.image)
    result = face_recogny.recogny_face(image)
    frame, bbox, name, current_time = result
    frame_b64 = RGB2String(result[0])
    return  name, current_time

@app.get("/tims/api/v1/checking")
async def hello():
    response = ExceptionType(200, "CC").getResponse()
    return response