import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from huggingface_hub import PyTorchModelHubMixin
config = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "max_length": 80,
    "csvfile": "/content/drive/MyDrive/toxic_comments.csv",
    "batch_size": 2,
    "learning_rate": 2e-5,
    "n_epochs": 1,
    "n_classes": 1,
    "device": torch.device("cuda" if torch.cuda.is_available else "cpu")

}

class CustomBertModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(CustomBertModel, self).__init__()
        self.pretrained_model = BertModel.from_pretrained(config['model_name'])   # bert base 768 hidden state
        self.classifier = nn.Linear(768, config['n_classes'])  # MLP

    def forward(self, input_ids, attention_mask):

        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)    # batch de 768
        output = self.classifier(output.last_hidden_state)

        return output