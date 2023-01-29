import time

from fastapi import FastAPI, Body
from transformers import pipeline
from typing import List
app = FastAPI()
start = time.time()
classifier = pipeline("zero-shot-classification", model=r"model/xlm-roberta-large-xnli")
print("Loaded model", time.time()-start)
@app.post("/predict")
async def predict(sequence_to_classify: str = Body(...), candidate_labels: List[str] = Body(...)):

    result = classifier(sequence_to_classify, candidate_labels)


    return {"scores": result}