import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
from KaggleAux import predict as ka # see github.com/agconti/kaggleaux for more details


df = pd.read_csv("data/train.csv") 

df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
df = df.dropna() 