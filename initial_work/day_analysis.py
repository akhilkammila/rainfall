# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
%load_ext autoreload
%autoreload 2

# %% Load data
DIRECTORY = "./data"
train = pd.read_csv(f"{DIRECTORY}/train.csv")
test = pd.read_csv(f"{DIRECTORY}/test.csv")



# %% Filter out wrong days
combined = pd.concat([train, test])
combined['day'] = combined.index % 365 + 1
combined = combined.reset_index(drop=True)

# %% Plot columns over time
columns_to_plot = [col for col in combined.columns if col not in ['id', 'day', 'rainfall']]
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=combined, x='id', y=column)
    plt.title(f'{column} over Time')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.show()

# %% Plot rolling average columns over time
columns_to_plot = [col for col in combined.columns if col not in ['id', 'day', 'rainfall']]
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=combined, x='id', y=combined[column].rolling(window=7, min_periods=1).mean())
    plt.title(f'Rolling average {column} over Time')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.show()

# %% Rainfall ewm
sns.scatterplot(data=combined, x='id', y=combined['rainfall'].ewm(alpha=0.1).mean())

# %%
combined


