import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from imports import import_func
import seaborn as sns


train = pd.read_csv("dataset/train.csv")

##Let us check how many columns have null values

train.columns[train.isnull().any()]

##Let Us check the % of missing values
missing_columns=train.isnull().sum()
missing_columns_percentage=missing_columns/len(train)
missing_columns_percentage=missing_columns_percentage[missing_columns_percentage>0].sort_values(ascending=False)

missing_frame=missing_columns_percentage.to_frame()

missing_frame.columns=['count']
missing_frame['Name']=missing_frame.index.values

sns.set(style="whitegrid", color_codes=True)
sns.barplot(missing_frame['Name'], missing_frame['count'])
plt.xticks(rotation = 90)


## Let us check the distribution of the Sale Price

sns.distplot(train['SalePrice'])

## It is positively or right skewed. Let us try the log value distribution
sns.distplot(np.log(train['SalePrice']))

## Yes the log form is normally distributed

#Let us split numeric and categorical data
numeric=train.select_dtypes(include=[np.number])
categoric=train.select_dtypes(exclude=[np.number])

corr_matrix=numeric.corr()

sns.heatmap(categoric, cmap="Blues")

correlation_saleprice=corr_matrix['SalePrice'].sort_values(ascending=False)

correlation_saleprice_frame=correlation_saleprice.to_frame()
correlation_saleprice_frame['Name']=correlation_saleprice_frame.index.values

sns.barplot(correlation_saleprice_frame['Name'], correlation_saleprice_frame['SalePrice'])
plt.xticks(rotation=90)