import torch
import gradio as gr


# Load model directly
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Fatou/Custom-Bert-Model")
model = AutoModel.from_pretrained("Fatou/Custom-Bert-Model")

classes = ["toxic", "no toxic"]

def predict(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')

        output = model(**inputs)
        pred = torch.max(output.logits, dim=1)
    return {"indice": pred.indices.item(),
            "classe": classes[pred.indices.item()]
            }


demo = gr.Interface(fn=predict, inputs="text", outputs="json")
demo.launch()   