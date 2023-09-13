from fastapi import FastAPI 
from pydantic import BaseModel
from utils import CustomBertModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from huggingface_hub import PyTorchModelHubMixin

app = FastAPI()    

model_loaded = CustomBertModel.from_pretrained("Fatou/Custom-Bert-Model")
tokenizer_loaded = AutoTokenizer.from_pretrained("Fatou/Custom-Bert-Model") 

classes = ["no toxic", "toxic"]

def predict(text):
    with torch.no_grad():
        inputs = tokenizer_loaded(text, return_tensors='pt')

        outputs = model_loaded(inputs['input_ids'], inputs['attention_mask'])
        pred = torch.max(torch.softmax(outputs, dim=1), dim=1)

    return {"indice": pred.indices.item(),
            "classe": classes[pred.indices.item()]
            }

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: InputText):
    text = data.text
    prediction_dict= predict(text)
    return prediction_dict


