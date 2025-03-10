# %% [markdown]
## Check how correlated sunshine, cloud, humidity are 

# %%
import pandas as pd
import numpy as np
import seaborn as sns


# %%
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# %%
sns.heatmap(train[['sunshine', 'cloud', 'humidity', 'rainfall']].corr(), annot=True, cmap='coolwarm')


# %%
