import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

# SUBMISSION_PATH = "submission.csv" #Path("submission.csv")
# FEATURES_PATH = "/content/drive/MyDrive/Mental Health Competition/Data/smoke_test_features.csv" #Path("data/test_features.csv")
# SUBMISSION_FORMAT_PATH = "/content/drive/MyDrive/Mental Health Competition/Data/submission_format.csv" #Path("data/submission_format.csv")
# MODEL_DIR = "/content/drive/MyDrive/Mental Health Competition/FT Models" #Path("assets")

SUBMISSION_PATH = Path("submission.csv")
FEATURES_PATH = Path("data/test_features.csv")
SUBMISSION_FORMAT_PATH = Path("data/submission_format.csv")

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits

def load_model(model_path, bert_model_name, num_classes, device):
    # Re-instantiate the tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Create a new model instance
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    # Load the model state dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model, tokenizer

# Load Models and Tokenizers
def load_models():
    models = []

    base_model = "bert-base-uncased"

    for i in range(2, 25): #25
        # print("Loading Model: ", i)
        model_path = f"./assets/models/bert_classifier_{i}.pth"
        if i == 23:
            model, tokenizer = load_model(model_path, base_model, num_classes=6, device="cuda")
        elif i == 24:
            model, tokenizer = load_model(model_path, base_model, num_classes=12, device="cuda")
        else:
            model, tokenizer = load_model(model_path, base_model, num_classes=2, device="cuda")
        models.append(model)

    return models



class TextClassificationDataset(Dataset):
    def __init__(self, ids, texts, tokenizer, max_length=128):
        self.ids = ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.ids[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'uids': ids, 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

def predict_for_single_model(model, model_num, input_ids, attention_mask, threshold=0.5):

    with torch.no_grad():
        # Get model outputs (logits)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)  # Use F.sigmoid for binary classification

        if model_num == 23 or model_num == 24:
            # For multiclass
            # Use argmax to get the predicted class for multi-class classification
            preds = probabilities.argmax(dim=1) + 1
        else:
            # Use the custom threshold to get predictions
            preds = (probabilities[:, 1] > threshold).long()  # Change '1' to the index of class `1`
    # print(f"Model {model_num} Predictions: {preds}")
    preds.cpu().tolist()
    torch.cuda.empty_cache()  # Clear memory after inference
    return preds

def inference_pipeline(features, models, tokenizer, device, submission_format, batch_size=16):
    # Combine features to create texts
    texts = (features['NarrativeCME'] + ' ' + features['NarrativeCME']).to_numpy()
    ids = features['uid'].to_numpy()

    # Create DataLoader object
    feature_dataset = TextClassificationDataset(ids, texts, tokenizer)
    dataloader = DataLoader(feature_dataset, batch_size=batch_size)

    # Initialize predictions with the same columns and index as submission_format
    predictions = pd.DataFrame(index=submission_format.index, columns=submission_format.columns)

    row_idx = 0  # To keep track of row position in the main DataFrame

    # Iterate through each batch first to keep tokenized inputs
    for idx, batch in enumerate(dataloader):
        # print("BATCH", idx)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        batch_preds = pd.DataFrame(index=submission_format.index[row_idx:row_idx + len(batch['input_ids'])])

        batch_preds['uid'] = batch['uids']

        # Iterate through all models for each batch
        for idx2, model in enumerate(models):
            preds = predict_for_single_model(model, idx2+2, input_ids, attention_mask)
            lst = preds.tolist()
            batch_preds[f'Column{idx2+1}'] = lst

        # display(batch_preds)

        # Concatenate batch predictions to the main predictions DataFrame
        predictions.loc[row_idx:row_idx + len(batch['input_ids']) - 1, :] = batch_preds.values
        row_idx += len(batch['input_ids'])

    # Ensure final column and row order matches submission_format
    # display(predictions)
    predictions = predictions[submission_format.columns]
    predictions = predictions.reindex(submission_format.index)

    # Remove rows with nan values
    predictions = predictions.dropna()

    # Ensure all values are integers for all columns except the first
    for col in predictions.columns[1:]:
        predictions[col] = predictions[col].round().astype(int)

    return predictions

def main():
    features = pd.read_csv(FEATURES_PATH)
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    # display(features)
    # display(submission_format)

    tokenizer = BertTokenizer.from_pretrained("./assets/bert-base-uncased-local")
    models = load_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = inference_pipeline(features, models, tokenizer, device, submission_format)
    predictions.to_csv(SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    main()