# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# %%
DIRECTORY = "../data"
train = pd.read_csv(f"{DIRECTORY}/train.csv")
test = pd.read_csv(f"{DIRECTORY}/test.csv")
train['day'] = train['id'] % 365 + 1
train['year'] = train['id']//365

# %% Guessing cloud overall
roc_auc_score(train['rainfall'], train['cloud']*train['humidity'])

# %%
# Initialize a list to store AUC scores for each year
auc_scores = []

for year in train['year'].unique():
    yearly_data = train[train['year'] == year]
    auc = roc_auc_score(yearly_data['rainfall'], yearly_data['cloud']*yearly_data['humidity'])
    auc_scores.append(auc)

    # Get fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(yearly_data['rainfall'], yearly_data['cloud'])

    # Plot the ROC curve for the year
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Year {year}, AUC = {auc:.4f}')
    plt.legend(loc="lower right")

# Print the AUC scores for each year
print("AUC scores for each year:", auc_scores)

# %% What are the "wrong" values when cloud is high?
wrong = (train['cloud'] > 95) & (train['rainfall'] == 0)
train.loc[wrong]

# %%
test['rainfall'] = test['cloud']*test['humidity']
test['rainfall'] = test['rainfall']/(test['rainfall'].max())
test[['id', 'rainfall']].to_csv(f"./test_cloud_humidity.csv", index=False)

# %%
