import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, AutoTokenizer
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import wandb   #for monitoring
import huggingface_hub
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

class MyDataset(Dataset):
    def __init__(self, csvfile, tokenizer_name, max_length):
        self.df = pd.read_csv(csvfile)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df['comment_text'][index]
        label = self.df['toxic'][index]

        inputs = self.tokenizer(text=text, max_length = self.max_length, padding = 'max_length', truncation =True, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'label': torch.tensor(label)

        }

def dataloader(dataset, batch_size, shuffle):

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

class CustomBertModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(CustomBertModel, self).__init__()
        self.pretrained_model = BertModel.from_pretrained(config['model_name'])   # bert base 768 hidden state
        self.classifier = nn.Linear(768, config['n_classes'])  # MLP

    def forward(self, input_ids, attention_mask):

        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)    # batch de 768
        output = self.classifier(output.last_hidden_state)

        return output


def train_step(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total = len(train_loader)):
        input_ids = data['input_ids'].squeeze(1).to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['label'].to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask)

        loss = loss_fn(output, label.unsqueeze(1))

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(train_loader)

def validation_step(model, validation_loader, loss_fn, device):

    total_loss = 0
    correct_prediction = 0

    with torch.no_grad():
        for data in tqdm(validation_loader, total=len(validation_loader)):
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label'].to(device)

            output = model(input_ids, attention_mask)

            loss = loss_fn(output, label.unsqueeze(1))

            pred = torch.max(torch.softmax(output, dim=1), dim=1)

            total_loss += loss.item()

            correct_prediction += torch.sum(pred.indices==label)

    return total_loss/len(validation_loader), 100*correct_prediction/len(validation_loader)



def main():

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    wandb.init(project="bert-classification")

    dataset = MyDataset(config['csvfile'], config['model_name'], config['max_length'])

    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2)

    train_loader = dataloader(train_dataset, config['batch_size'], shuffle = True)

    validation_loader = dataloader(validation_dataset, config['batch_size'], shuffle = False)

    data = next(iter(train_loader))

    model = CustomBertModel(config['model_name'], n_classes = 1)

    model.to(config['device'])

    #output = model(data['input_ids'].squeeze(1), data['attention_mask'])

    optimizer = AdamW(model.parameters(), lr = config['learning_rate'])

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config['n_epochs']):
        loss_train = train_step(model, train_loader, optimizer, loss_fn, config['device'])
        loss_validation, accuracy = validation_step(model, validation_loader, loss_fn, config['device'])

        wandb.log({
            "loss_train": loss_train,
            "loss_validation": loss_validation,
            "accuracy": accuracy
        })


    # Push model to the Hub (Note that we should login before pushing by using the command huggingface-cli login)
    model.push_to_hub("Fatou/Custom-Bert-Model")
    tokenizer.push_to_hub("Fatou/Custom-Bert-Model")
    #model.pretrained_model.config.push_to_hub("Fatou/Custom-Bert-Model")

    print("Congrats, your pytorch model and the tokenizer have been pushed successfully")

    # What follows could be out of the method main()

    # This is a way to load the model and its tokenizer
    model_loaded = CustomBertModel.from_pretrained("Fatou/Custom-Bert-Model")
    tokenizer_loaded = AutoTokenizer.from_pretrained("Fatou/Custom-Bert-Model")

    # This is a way to make predictions from the loaded model and tokenizer
    inputs = tokenizer_loaded("Hello, my dog is cute", return_tensors="pt")
    outputs = model_loaded(inputs['input_ids'], inputs['attention_mask'])
    pred = torch.max(torch.softmax(outputs, dim=1), dim=1)
    pred.indices.item()

if __name__ == '__main__':
    main()