import os
import shutil
import sys
import pandas as pd

df = pd.read_json("Dataset/FaceData/processed/data_face.json")


def main(email):
    members = df[df["email"] != email]
    members.to_json("Dataset/FaceData/processed/data_face.json")
    shutil.rmtree("Dataset/FaceData/raw/" + email)
    # os.remove('Dataset/FaceData/raw/' + name)


if __name__ == "__main__":
    email = sys.argv[1]
    sys.exit(main(email) or 0)
