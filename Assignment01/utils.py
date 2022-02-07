#%%
import pandas as pd
data1 = pd.read_csv('iris.data', header=0, names=['sepl','sepw','petl','petw','class'])
# print(data)
data1.describe()


data2 = pd.read_json('hepatitis_training_data.json')
data2.describe()