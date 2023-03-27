# Main.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import numpy as np
from PIL import Image
from typing import Union

# Package
from training.src.make_data_train import Trainer
from training.src.face_rec import Face_recognition
from .helpers.base64_convert import stringToRGB, RGB2String
from .helpers.exception_handler import ExceptionType, exception_types
from .mocks.response import data, error

# Init application
face_recogny = Face_recognition()
app = FastAPI(exception_handlers = exception_types)


# Cors-header-origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# Class 
class Member(BaseModel):
    name: str
    image: str

class Detection(BaseModel):
    image: str

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

# Router
@app.post("/tims/api/v1/checking/user")
async def add_new_member(item: Member):
    trainer = Trainer()
    img = stringToRGB(item.image)
    trainer.add_member(image = img, name = item.name)
    response = ExceptionType(200, "Success", data = data).getResponse()
    return JSONResponse(status_code = 200, content = response)

@app.delete("/tims/api/v1/checking/user/{name}")
async def delete_member(name: str):
    response = ExceptionType(200, "Success", data = data).getResponse()
    return JSONResponse(status_code = 200, content = response)

@app.post("/tims/api/v1/checking")
async def checking(item: Detection):
    image = stringToRGB(item.image)
    result = face_recogny.recogny_face(image)
    if result == None:
        response = ExceptionType(400, "Bad Request", error = error).getResponse()
        return JSONResponse(status_code = 400, content = response)
    frame, bbox, name, current_time = result
    frame_b64 = RGB2String(result[0])
    if name == "Unknown":
        response = ExceptionType(400, "Bad Request", error = error).getResponse()
        return JSONResponse(status_code = 400, content = response)
    else:
        response = ExceptionType(200, "Success", data = data(name)).getResponse()
        return JSONResponse(status_code = 200, content = response)

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return None

@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item

@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Union[str, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

# Running
if __name__ == "__main__":
    uvicorn.run(app)