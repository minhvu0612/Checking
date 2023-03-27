import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from .face_identification import Face_identifier
from .face_detection import Face_detector
import pandas as pd
import numpy as np
from .settings import (
    DATA_FACE_DIR,
)
import shutil
class Trainer:
    "Make dataset - CRUD"
    def __init__(self):
        self.df = pd.read_json(DATA_FACE_DIR)
        self.data_image = self.df["face"].to_numpy().tolist()
        self.members = self.df["email"].to_numpy().tolist()
        self.face_identifier = Face_identifier()
        self.face_detector = Face_detector()

    def add_member(self, image, email):
        "Add new menber data into dataset"
        image = self.face_detector.detect_face(image)[0]
        if image is None:
            return None
        image = image[0]
        image_embedding = self.face_identifier.embed_image(image)
        self.data_image.append(image_embedding)
        self.members.append(email)
        np.append(self.members, email)
        df = pd.DataFrame({"face": self.data_image, "email": self.members})
        df.to_json(DATA_FACE_DIR)
        return "successfully"
    
    def delete_member(self, email):
        members = self.df[self.df["email"] != email]
        members.to_json(DATA_FACE_DIR)
        shutil.rmtree("datasets/face/raw/" + email)
