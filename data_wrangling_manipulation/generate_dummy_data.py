# Generating a dummy data frame
import pandas as pd
import numpy as np
import os

# set random state
np.random.seed(1)

df = pd.DataFrame({'a': np.random.randn(100), 'b': 2.5*np.random.randn(100) + 1.5})

print(os.getcwd())
data = pd.read_csv('Customer data.csv')

columns = data.columns.values.tolist()
df = pd.DataFrame({'a': np.random.randn(len(columns)), 
                   'b': 2.5*np.random.randn(len(columns)) + 1.5,
                  'Column_names': columns})
print(df)
