import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


df = pd.read_csv("dataset/trainingdata.txt")

df=df[df['charge']!=8]

X_train=df.iloc[:, 0]
Y=df.iloc[:, 1]



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train.reshape(-1, 1), Y.reshape(-1, 1))
K=lr.predict(6)