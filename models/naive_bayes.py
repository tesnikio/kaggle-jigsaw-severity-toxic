import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0 ).astype(int)
df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
min_len = (df['y'] == 1).sum()
df_y0_undersample = df[df['y'] == 0].sample(n=min_len, random_state=201)
df = pd.concat([df[df['y'] == 1], df_y0_undersample])
vec = TfidfVectorizer()
X = vec.fit_transform(df['text'])
model = MultinomialNB()
model.fit(X, df['y'])
df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
X_test = vec.transform(df_sub['text'])
preds6 = model.predict_proba(X_test)[:, 1]