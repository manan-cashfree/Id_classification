from pydantic import BaseModel
from enum import Enum
from typing import List
from fastapi import FastAPI, UploadFile
from src.predict import get_model, predict

app = FastAPI()
model = get_model()


class Classes(str, Enum):
    aadhaar_front = 'Aadhaar Front'
    aadhaar_back = 'Aadhaar back'
    invalid = 'Invalid'
    pan = 'PAN'


class PredictSchema(BaseModel, use_enum_values=True):
    Class: Classes
    score: float
    tags: List = []


@app.post("/v1/predict/")
def get_document_type(img_file: UploadFile) -> PredictSchema:
    pred, conf = predict(model, img_file.file)
    return PredictSchema(
        Class=pred,
        score=conf
    )
