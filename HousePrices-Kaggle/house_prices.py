import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from imports import import_func

df = pd.read_csv("dataset/train.csv")

Y=df['SalePrice']

df=df.drop(['SalePrice'], axis=1)

empty_columns=check_empty_value_columns(df)