import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
import time
from .face_detection import Face_detector
from .face_identification import Face_identifier
import cv2
import pandas as pd
import numpy as np
from .settings import (
    DATA_FACE_DIR,
)


face_detector = Face_detector()
face_identifier = Face_identifier()
df = pd.read_json(DATA_FACE_DIR)
data_faces = df["face"].to_numpy().tolist()
members = df["name"].to_numpy().tolist()


class Face_recognition:
    ''' Face recognition class '''
    def __init__(self) -> None:
        self.df = pd.read_json(DATA_FACE_DIR)
        self.data_faces = self.df["face"].to_numpy().tolist()
        self.members = self.df["name"].to_numpy().tolist()
        self.face_detector = Face_detector()
        self.face_identifier = Face_identifier()

    def recogny_face(self, image: np.ndarray):
        imgs, x, y  = self.face_detector.detect_face(image)
        if len(imgs) == 0:
            return None
        for i in range(len(imgs)):
            xmin, xmax = x[i]
            ymin, ymax = y[i]
            self.bbox = [[xmin, ymin], [xmax, ymax]]
            self.frame = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            self.name_member = self.face_identifier.result_name(imgs[i], self.data_faces, self.members)
        self.current_time = datetime.now()
        return self.frame, self.bbox, self.name_member, self.current_time


def main(image):
    imgs, x, y = face_detector.detect_face(image)
    if len(imgs) == 0:
        return None
    for i in range(len(imgs)):
        xmin, xmax = x[i]
        ymin, ymax = y[i]
        bbox = [[xmin, ymin], [xmax, ymax]]
        frame = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        name_member = face_identifier.result_name(imgs[i], data_faces, members)
        if name_member != "Unknown":
            current_time = datetime.now()
            break
        current_time = datetime.now()
    return frame, bbox, name_member, current_time


if __name__ == "__main__":
    image = cv2.imread('ll2.png')
    print(main(image))
