import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imports import import_func
from sklearn.preprocessing import Imputer

# Importing the dataset]
dataset_b = pd.read_csv('dataset/train/train.csv')
dataset = dataset_b
empty_columns=check_empty_value_columns(dataset)

value_mean=pd.DataFrame(dataset.iloc[:, column]).mean()
count = 0
dataset["Cabin"]=dataset["Cabin"].fillna(0)
while count < len(empty_columns):
    column=dataset.columns.get_loc(empty_columns[count])
    #dataset.iloc[:, dataset.columns.get_loc(empty_columns[count])]=fill_empty_values_in_column(dataset, dataset.columns.get_loc(empty_columns[count]), 'mean')
    
    value_mean=pd.DataFrame(dataset.iloc[:, column]).mean()
    count += 1  


