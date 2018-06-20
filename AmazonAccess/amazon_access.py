
# coding: utf-8

# In[48]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer
import random

import seaborn as sns

from imports import categorical_encoder
from imports import dataframe_selector
from imports import categorical_imputer

# Importing the dataset]
dataset_b = pd.read_csv('data/train.csv')
train_dataset= dataset_b.copy()



train_dataset




X=train_dataset.iloc[:, 1:11]




Y=train_dataset.iloc[:, 0]



Y.head()


X



dataset_c = pd.read_csv('data/test.csv')
test_dataset= dataset_c.copy()




X_test=test_dataset.iloc[:, 1:11]





X['RESOURCE'].value_counts()



X2=pd.get_dummies(X,drop_first=True)



T=X



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X.iloc[:, 0] = labelencoder_X_0.fit_transform(X.iloc[:, 0])




X.head()



onehotencoder = OneHotEncoder(categorical_features = "all")
X = onehotencoder.fit_transform(X).toarray()


onehotencoder_test = OneHotEncoder(categorical_features = "all")
X_test = onehotencoder_test.fit_transform(X_test).toarray()

