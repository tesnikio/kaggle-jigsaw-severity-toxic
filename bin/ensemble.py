df = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
df['model1'] = preds1
df['model2'] = preds2
df['model3'] = preds3
df['model4'] = preds4
df['model5'] = preds5
df['model6'] = preds6
df['model7'] = preds7

cols = [c for c in df.columns if c.startswith('model')]

# Put all predictions in the same scale.
# Make all the distances between predictions uniform
df[cols] = df[cols].rank(method='first').astype(int)

# Weights of each model
weights = {
    'model1': 0.35, # 0.816
    'model2': 0.43, # 0.825
    'model3': 0.06, # 0.807
    'model4': 0.06, # 0.806
    'model5': 0.02, # 0.782
    'model6': 0.02, # 0.768
    'model7': 0.06  # 0.812
}

# A weighted sum determines the final position
# It is the same as an average in the end
df['score'] = pd.DataFrame([df[c] * weights[c] for c in cols]).T.sum(axis=1).rank(method='first').astype(int)
df.head()