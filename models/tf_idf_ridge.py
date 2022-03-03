import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import scipy
pd.options.display.max_colwidth=300

df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

# Give more weight to severe toxic
df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})


df = pd.concat([df[df.y>0] ,
                df[df.y==0].sample(int(len(df[df.y>0])*1.5)) ], axis=0).sample(frac=1)

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        ("clf", Ridge()),
    ]
)
pipeline.fit(df['text'], df['y'])
df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
preds7 = pipeline.predict(df_sub['text'])