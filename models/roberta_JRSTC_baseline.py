import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import re
import random
import gc
import glob

pd.set_option('display.max_columns', None)
np.seterr(divide='ignore', invalid='ignore')
gc.enable()

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
# NLP
from transformers import AutoTokenizer, AutoModel

# Random Seed Initialize
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

data_dir = '../input/jigsaw-toxic-severity-rating'
models_dir = '../input/jrstc-models/roberta_base'
test_file_path = os.path.join(data_dir, 'comments_to_score.csv')
print(f'Train file: {test_file_path}')

test_df = pd.read_csv(test_file_path)


# Text Cleaning

def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


tqdm.pandas()
test_df['text'] = test_df['text'].progress_apply(text_cleaning)

test_df.sample(10)

# CFG

params = {
    'device': device,
    'debug': False,
    'checkpoint': '../input/roberta-base',
    'output_logits': 768,
    'max_len': 256,
    'batch_size': 32,
    'dropout': 0.2,
    'num_workers': 2
}

if params['debug']:
    train_df = train_df.sample(frac=0.01)
    print('Reduced training Data Size for Debugging purposes')


# Dataset

class BERTDataset:
    def __init__(self, text, max_len=params['max_len'], checkpoint=params['checkpoint']):
        self.text = text
        self.max_len = max_len
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.num_examples = len(self.text)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        text = str(self.text[idx])

        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = tokenized_text['input_ids']
        mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}


# NLP Model

class ToxicityModel(nn.Module):
    def __init__(self, checkpoint=params['checkpoint'], params=params):
        super(ToxicityModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(params['output_logits'], 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(params['dropout']),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds


# Prediction

predictions_nn = None
for model_name in glob.glob(models_dir + '/*.pth'):
    model = ToxicityModel()
    model.load_state_dict(torch.load(model_name))
    model = model.to(params['device'])
    model.eval()

    test_dataset = BERTDataset(
        text=test_df['text'].values
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params['batch_size'],
        shuffle=False, num_workers=params['num_workers'],
        pin_memory=True
    )

    temp_preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Predicting. '):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            predictions = model(ids, token_type_ids, mask).to('cpu').numpy()

            if temp_preds is None:
                temp_preds = predictions
            else:
                temp_preds = np.vstack((temp_preds, predictions))

    if predictions_nn is None:
        predictions_nn = temp_preds
    else:
        predictions_nn += temp_preds

predictions_nn /= (len(glob.glob(models_dir + '/*.pth')))

preds3 = predictions_nn
preds3 = preds3.squeeze(-1)