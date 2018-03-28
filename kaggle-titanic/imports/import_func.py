import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def check_empty_value_columns(dataset):
    return dataset.columns[dataset.isna().any()].tolist();

def fill_empty_values_in_column(dataframe, column, filler):
   
    if(filler=='mean'):
          value_mean=pd.DataFrame(dataframe.iloc[:, column]).mean()
          print(value_mean)
          return (pd.DataFrame(dataframe.iloc[:, column]).fillna(value_mean[0]).iloc[:, :])
    