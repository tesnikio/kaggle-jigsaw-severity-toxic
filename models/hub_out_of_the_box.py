import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    """
    For comments_to_score.csv (the submission), get only one comment per row
    """

    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long)
        }


def generate_predictions(model_path, max_len, is_multioutput):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to("cuda")
    model.eval()

    df = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")

    dataset = Dataset(text=df.text.values, tokenizer=tokenizer, max_len=max_len)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, num_workers=2, pin_memory=True, shuffle=False
    )

    final_output = []

    for data in data_loader:
        with torch.no_grad():
            for key, value in data.items():
                data[key] = value.to("cuda")
            output = model(**data)

            if is_multioutput:
                # Sum the logits for all the toxic labels
                # One strategy out of various possible
                output = output.logits.sum(dim=1)
            else:
                # Classifier. Get logits for "toxic"
                output = output.logits[:, 1]

            output = output.detach().cpu().numpy().tolist()
            final_output.extend(output)

    torch.cuda.empty_cache()
    return np.array(final_output)


preds_bert = generate_predictions("../input/toxic-bert", max_len=192, is_multioutput=True)
preds_rob1 = generate_predictions("../input/roberta-base-toxicity", max_len=192, is_multioutput=False)
preds_rob2 = generate_predictions("../input/roberta-toxicity-classifier", max_len=192, is_multioutput=False)

df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
df_sub["score_bert"] = preds_bert
df_sub["score_rob1"] = preds_rob1
df_sub["score_rob2"] = preds_rob2
df_sub[["score_bert", "score_rob1", "score_rob2"]] = MinMaxScaler().fit_transform(
    df_sub[["score_bert", "score_rob1", "score_rob2"]])

preds5 = df_sub[["score_bert", "score_rob1", "score_rob2"]].sum(axis=1)