path = "training/"

MODEL_DETECT_FACE_XML = (
    path + "models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
)
MODEL_DETECT_FACE_BIN = (
    path + "models/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"
)

MODEL_IDENTIFY_XML = path + "models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
MODEL_IDENTIFY_BIN = path + "models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin"

FACE_SIZE = 128

DATA_FACE_DIR   = path + "datasets/face/processed/data_face.json"
DATA_RESULT_DIR = path + "datasets/face/processed/result.json"
