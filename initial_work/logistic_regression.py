# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data
DIRECTORY = "./data"
train = pd.read_csv(f"{DIRECTORY}/train.csv")
test = pd.read_csv(f"{DIRECTORY}/test.csv")
train['day'] = train.index % 365 + 1

# %% Clean data
def clean_data(df):
    chosen_cols = ['id', 'sunshine', 'cloud', 'humidity']
    if 'rainfall' in df.columns: chosen_cols.append('rainfall')
    df = df[chosen_cols].bfill()
    return df
train = clean_data(train)
test = clean_data(test)


# %% Prepare features and target variable
X = train.drop(columns=['rainfall'])  # Features
y = train['rainfall']  # Target variable

# %% Set up logistic regression model
model = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

# %% Print cross-validation results
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# %% Fit the model on the training data
model.fit(X, y)
# predictions = model.predict_proba(test)[:, 1]
predictions = 

# %% Output predictions
print("Predictions on the test set:", predictions)

# Optionally, save predictions to a CSV file
output = pd.DataFrame({'id': test['id'], 'rainfall': predictions})
output['rainfall'] = output['rainfall'].apply(lambda x: 1 if x >= 0.9 else 0 if x <= 0.1 else x)

output.to_csv('test_predictions.csv', index=False)

