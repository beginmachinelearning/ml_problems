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
dataset_b = pd.read_csv('dataset/train/train.csv')
train_dataset= dataset_b.copy()

train_dataset['AgeRange']=train_dataset['Age']/20
train_dataset['AgeRange']=train_dataset['AgeRange'].fillna(train_dataset['AgeRange'].mean())
train_dataset['AgeRange']=train_dataset['AgeRange'].astype(int)


train_dataset['Family']=train_dataset['SibSp']+train_dataset['Parch']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare", "AgeRange", "Family"])),
        ("imputer", Imputer(strategy="median")),
    ])
    
T=num_pipeline.fit_transform(train_dataset)    


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
    ])


cat_pipeline.fit_transform(train_dataset)

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    
X_train = preprocess_pipeline.fit_transform(train_dataset)    

y_train=train_dataset['Survived']


from sklearn.svm import SVC
svc_reg=SVC()
svc_reg.fit(X_train, y_train)


from sklearn.model_selection import cross_val_score

accuracy=cross_val_score(svc_reg, X_train, y_train, cv=10)
accuracy.mean()


from sklearn.tree import DecisionTreeRegressor
regressordt = DecisionTreeRegressor(random_state = 0)
regressordt.fit(X_train, y_train)



from sklearn.ensemble import RandomForestClassifier
regressorrf = RandomForestClassifier(n_estimators = 10, random_state = 0)
regressorrf.fit(X_train, y_train)

accuracy=cross_val_score(regressorrf, X_train, y_train, cv=10)
accuracy.mean()


dataset_test = pd.read_csv('dataset/test/test.csv')
dataset_test_b= dataset_test.copy()

dataset_test['AgeRange']=dataset_test['Age']/20
dataset_test['AgeRange']=dataset_test['AgeRange'].fillna(dataset_test['AgeRange'].mean())
dataset_test['AgeRange']=dataset_test['AgeRange'].astype(int)


dataset_test['Family']=dataset_test['SibSp']+dataset_test['Parch']

X_test = preprocess_pipeline.fit_transform(dataset_test) 



ypred=regressorrf.predict(X_test)


from sklearn.svm import SVC
svc_reg=SVC()
svc_reg.fit(X_train, y_train)

ypred=svc_reg.predict(X_test)
