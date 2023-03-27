# Encode to base64 and decode from base64
import numpy as np
import base64
import cv2


def stringToRGB(base64_string):
    '''Decode base64 string to image'''
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def RGB2String(image):
    '''Encode image to base64'''
    _, im_arr = cv2.imencode('.jpg', image)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64