# %% [markdown]
# Feature Magnitudes
## 1. Guessing based on how extreme the values are (what percentage of rows have values that extreme w/ and w/o rain))

# %%
import pandas as pd
import numpy as np



# %%
# Load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# %%
# Calculate the percentage of rows with extreme values

