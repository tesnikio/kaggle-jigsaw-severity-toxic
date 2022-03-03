import os
import gc
import copy
import time
import random
import string

import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re
from nltk.corpus import stopwords

from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler


class Config:
    num_classes = 1
    epochs = 10
    margin = 0.5
    model_name = '../input/robertalarge'
    batch_size = 8
    lr = 1e-4
    weight_decay = 0.01
    scheduler = 'CosineAnnealingLR'
    max_length = 196
    accumulation_step = 1
    patience = 1


class ToxicDataset(Dataset):
    def __init__(self, comments, tokenizer, max_length):
        self.comment = comments
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        inputs_more_toxic = self.tokenizer.encode_plus(
            self.comment[idx],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        input_ids = inputs_more_toxic['input_ids']
        attention_mask = inputs_more_toxic['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class ToxicModel(nn.Module):
    def __init__(self, model_name, args):
        super(ToxicModel, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(1024, self.args.num_classes)

    def forward(self, toxic_ids, toxic_mask):
        out = self.model(
            input_ids=toxic_ids,
            attention_mask=toxic_mask,
            output_hidden_states=False
        )

        out = self.dropout(out[1])
        outputs = self.output(out)

        return outputs


def get_predictions(model, dataloader):
    model.eval()

    PREDS = []
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()

            outputs = model(input_ids, attention_mask)

            PREDS.append(outputs.view(-1).cpu().detach().numpy())

            bar.set_postfix(Stage='Inference')

        PREDS = np.hstack(PREDS)
        gc.collect()

        return PREDS


df = pd.read_csv('../input/jigsaw-toxic-severity-rating/validation_data.csv')

args = Config()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


def washing_machine(comments):
    corpus = []
    for i in tqdm(range(len(comments))):
        comment = re.sub('[^a-zA-Z]', ' ', comments[i])
        comment = comment.lower()
        comment = comment.split()
        stemmer = SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()
        all_stopwords = stopwords.words('english')
        comment = [stemmer.stem(word) for word in comment if not word in set(all_stopwords)]
        comment = [lemmatizer.lemmatize(word) for word in comment]
        comment = ' '.join(comment)
        corpus.append(comment)

    return corpus


def inference(dataloader):
    final_preds = []
    args = Config()
    base_path = '../input/large-jigsaw-kishal-v1/'
    for fold in range(1):
        model = ToxicModel(args.model_name, args)
        model = model.cuda()
        path = base_path + f'model_fold_0.bin'
        model.load_state_dict(torch.load(path))

        print(f"Getting predictions for model {fold + 1}")
        preds = get_predictions(model, dataloader)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


# Prediction and submission

sub = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')

sub.head(1)

sub_dataset = ToxicDataset(washing_machine(sub['text'].values), tokenizer, max_length=args.max_length)
sub_loader = DataLoader(sub_dataset, batch_size=2 * args.batch_size,
                        num_workers=2, shuffle=False, pin_memory=True)

preds4 = inference(sub_loader)