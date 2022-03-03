import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata


def ridge_cv(vec, X, y, X_test, folds, stratified):
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=12)
    val_scores = []
    rmse_scores = []
    X_less_toxics = []
    X_more_toxics = []

    preds = []
    for fold, (train_index, val_index) in enumerate(kf.split(X, stratified)):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        model = Ridge()
        model.fit(X_train, y_train)

        rmse_score = mean_squared_error(model.predict(X_val), y_val, squared=False)
        rmse_scores.append(rmse_score)

        X_less_toxic = vec.transform(df_val['less_toxic'])
        X_more_toxic = vec.transform(df_val['more_toxic'])

        p1 = model.predict(X_less_toxic)
        p2 = model.predict(X_more_toxic)

        X_less_toxics.append(p1)
        X_more_toxics.append(p2)

        # Validation Accuracy
        val_acc = (p1 < p2).mean()
        val_scores.append(val_acc)

        pred = model.predict(X_test)
        preds.append(pred)

        print(f"FOLD:{fold}, rmse_fold:{rmse_score:.5f}, val_acc:{val_acc:.5f}")

    mean_val_acc = np.mean(val_scores)
    mean_rmse_score = np.mean(rmse_scores)

    p1 = np.mean(np.vstack(X_less_toxics), axis=0)
    p2 = np.mean(np.vstack(X_more_toxics), axis=0)

    val_acc = (p1 < p2).mean()

    print(f"OOF: val_acc:{val_acc:.5f}, mean val_acc:{mean_val_acc:.5f}, mean rmse_score:{mean_rmse_score:.5f}")

    preds = np.mean(np.vstack(preds), axis=0)

    return p1, p2, preds


toxic = 1.0
severe_toxic = 2.0
obscene = 1.0
threat = 1.0
insult = 1.0
identity_hate = 2.0


def create_train(df):
    df['y'] = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].max(axis=1)
    df['y'] = df["y"] + df['severe_toxic'] * severe_toxic
    df['y'] = df["y"] + df['obscene'] * obscene
    df['y'] = df["y"] + df['threat'] * threat
    df['y'] = df["y"] + df['insult'] * insult
    df['y'] = df["y"] + df['identity_hate'] * identity_hate

    df = df[['comment_text', 'y', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].rename(
        columns={'comment_text': 'text'})

    # undersample non toxic comments  on Toxic Comment Classification Challenge
    min_len = (df['y'] >= 1).sum()
    df_y0_undersample = df[df['y'] == 0].sample(n=int(min_len * 1.5), random_state=201)
    df = pd.concat([df[df['y'] >= 1], df_y0_undersample])

    return df


df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
df_test = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")

jc_train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
print(f"jc_train_df:{jc_train_df.shape}")

jc_train_df = create_train(jc_train_df)

df = jc_train_df
print(df['y'].value_counts())

FOLDS = 7

vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6))
X = vec.fit_transform(df['text'])
y = df["y"].values
X_test = vec.transform(df_test['text'])

stratified = np.around(y)
jc_p1, jc_p2, jc_preds = ridge_cv(vec, X, y, X_test, FOLDS, stratified)

juc_train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
print(f"juc_train_df:{juc_train_df.shape}")
juc_train_df = juc_train_df.query("toxicity_annotator_count > 5")
print(f"juc_train_df:{juc_train_df.shape}")

juc_train_df['y'] = juc_train_df[
    ['severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']].sum(axis=1)

juc_train_df['y'] = juc_train_df.apply(lambda row: row["target"] if row["target"] <= 0.5 else row["y"], axis=1)
juc_train_df = juc_train_df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
min_len = (juc_train_df['y'] > 0.5).sum()
df_y0_undersample = juc_train_df[juc_train_df['y'] <= 0.5].sample(n=int(min_len * 1.5), random_state=201)
juc_train_df = pd.concat([juc_train_df[juc_train_df['y'] > 0.5], df_y0_undersample])

df = juc_train_df
print(df['y'].value_counts())

FOLDS = 7

vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6))
X = vec.fit_transform(df['text'])
y = df["y"].values
X_test = vec.transform(df_test['text'])

stratified = (np.around(y, decimals=1) * 10).astype(int)
juc_p1, juc_p2, juc_preds = ridge_cv(vec, X, y, X_test, FOLDS, stratified)

rud_df = pd.read_csv("../input/ruddit-jigsaw-dataset/Dataset/ruddit_with_text.csv")
print(f"rud_df:{rud_df.shape}")
rud_df['y'] = rud_df['offensiveness_score'].map(lambda x: 0.0 if x <= 0 else x)
rud_df = rud_df[['txt', 'y']].rename(columns={'txt': 'text'})
min_len = (rud_df['y'] < 0.5).sum()
print(rud_df['y'].value_counts())

FOLDS = 7
df = rud_df
vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6))
X = vec.fit_transform(df['text'])
y = df["y"].values
X_test = vec.transform(df_test['text'])

stratified = (np.around(y, decimals=1) * 10).astype(int)
rud_p1, rud_p2, rud_preds = ridge_cv(vec, X, y, X_test, FOLDS, stratified)

jc_max = max(jc_p1.max(), jc_p2.max())
juc_max = max(juc_p1.max(), juc_p2.max())
rud_max = max(rud_p1.max(), rud_p2.max())

p1 = jc_p1 / jc_max + juc_p1 / juc_max + rud_p1 / rud_max
p2 = jc_p2 / jc_max + juc_p2 / juc_max + rud_p2 / rud_max

val_acc = (p1 < p2).mean()
print(f"Ensemble: val_acc:{val_acc:.5f}")

preds2 = jc_preds / jc_max + juc_preds / juc_max + rud_preds / rud_max