import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imports import import_func
from sklearn.preprocessing import Imputer
import random

import seaborn as sns
# Importing the dataset]
dataset_b = pd.read_csv('dataset/train/train.csv')
dataset= dataset_b.copy()

#Find Empty columns
empty_columns=check_empty_value_columns(dataset)

#Find datatype
dataset_b.info()

#Mean/Variance
dataset_b.describe()

#Check survival by class
dataset_b[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Check survival by Sex
dataset_b[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

dataset_b[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

dataset_b[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(dataset_b, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(dataset_b, col='Survived')
g.map(plt.hist, 'Sex', bins=2)


dataset['desig'] = dataset['Name'].str.extract(' ([a-zA-Z]+)\.')
dataset['desig'].unique()
dataset['desig'] = dataset['desig'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['desig'] = dataset['desig'].replace('Mlle', 'Miss')
dataset['desig'] = dataset['desig'].replace('Ms', 'Miss')
dataset['desig'] = dataset['desig'].replace('Mme', 'Mrs')

#Check survival by designation
dataset[['desig', 'Survived']].groupby(['desig'], as_index=False).mean().sort_values(by='Survived', ascending=False)

dataset['desig'] = dataset['desig'].replace('Mr', '1')
dataset['desig'] = dataset['desig'].replace('Rare', '2')
dataset['desig'] = dataset['desig'].replace('Master', '3')
dataset['desig'] = dataset['desig'].replace('Miss', '4')
dataset['desig'] = dataset['desig'].replace('Mrs', '5')




dataset["Cabin"]=dataset["Cabin"].fillna(0)
dataset["Cabin"] = np.where(dataset["Cabin"]==0, 0, 1)

dataset["Age"]=dataset["Age"].fillna(0)
dataset["Age"] = np.where(dataset["Age"]==0, random.randint(1,70), dataset["Age"])


dataset.loc[dataset['Age'] <= 20, 'AgeBand'] = 1
dataset.loc[(dataset["Age"]>20) & (dataset["Age"]<=40) , 'AgeBand'] = 2
dataset.loc[(dataset["Age"]>40) & (dataset["Age"]<=60) , 'AgeBand'] = 3
dataset.loc[(dataset["Age"]>60) & (dataset["Age"]<=80) , 'AgeBand'] = 4

dataset[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)

dataset = dataset.drop(['Name', 'PassengerId', 'Ticket' ], axis=1)


dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].value_counts().index[0])

X = dataset.iloc[:, :].values
y = dataset.iloc[:, 0].values

X = np.delete(X, 0, axis=1)

df=pd.DataFrame(X)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1] )

labelencoder_embark = LabelEncoder()
X[:, 7] = labelencoder_embark.fit_transform(X[:, 7] )


df=pd.DataFrame(X)



df=pd.DataFrame(X)

onehotencoder_embarked = OneHotEncoder(categorical_features = [1])
X = onehotencoder_embarked.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]

df=pd.DataFrame(X)

onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X).toarray()

df=pd.DataFrame(X)

X = X[:, 1:]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = np.round(regressor.predict(X))

regressor.score(X, y)

from sklearn.ensemble import RandomForestRegressor
regressorrf = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorrf.fit(X, y)

regressorrf.score(X, y)

from sklearn.tree import DecisionTreeRegressor
regressordt = DecisionTreeRegressor(random_state = 0)
regressordt.fit(X, y)

regressordt.score(X, y)

dataset_test = pd.read_csv('dataset/test/test.csv')
dataset_t = dataset_test.copy()
empty_columns=check_empty_value_columns(dataset_t)


dataset_t['desig'] = dataset_t['Name'].str.extract(' ([a-zA-Z]+)\.')
dataset_t['desig'].unique()
dataset_t['desig'] = dataset_t['desig'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset_t['desig'] = dataset_t['desig'].replace('Mlle', 'Miss')
dataset_t['desig'] = dataset_t['desig'].replace('Ms', 'Miss')
dataset_t['desig'] = dataset_t['desig'].replace('Mme', 'Mrs')


dataset_t['desig'] = dataset_t['desig'].replace('Mr', '1')
dataset_t['desig'] = dataset_t['desig'].replace('Rare', '2')
dataset_t['desig'] = dataset_t['desig'].replace('Master', '3')
dataset_t['desig'] = dataset_t['desig'].replace('Miss', '4')
dataset_t['desig'] = dataset_t['desig'].replace('Mrs', '5')




dataset_t["Cabin"]=dataset_t["Cabin"].fillna(0)
dataset_t["Cabin"] = np.where(dataset_t["Cabin"]==0, 0, 1)

dataset_t["Age"]=dataset_t["Age"].fillna(0)
dataset_t["Age"] = np.where(dataset_t["Age"]==0, random.randint(1,70), dataset_t["Age"])

dataset_t["AgeBand"]=''

dataset_t.loc[dataset_t['Age'] <= 20, 'AgeBand'] = 1
dataset_t.loc[(dataset_t["Age"]>20) & (dataset_t["Age"]<=40) , 'AgeBand'] = 2
dataset_t.loc[(dataset_t["Age"]>40) & (dataset_t["Age"]<=60) , 'AgeBand'] = 3
dataset_t.loc[(dataset_t["Age"]>60) & (dataset_t["Age"]<=80) , 'AgeBand'] = 4


value_mean=pd.DataFrame(dataset_t.iloc[:, 8]).mean()
dataset_t["Fare"]=dataset_t["Fare"].fillna(value_mean[0])


X_test = dataset_t.iloc[:, :].values

X_test = np.delete(X_test, 2, axis=1)
X_test = np.delete(X_test, 6, axis=1)


X_test = np.delete(X_test, 3, axis=1)

df=pd.DataFrame(X_test)

labelencoder_test = LabelEncoder()
X_test[:, 2] = labelencoder_test.fit_transform(X_test[:, 2] )

labelencoder_test_embark = LabelEncoder()
X_test[:, 8] = labelencoder_test_embark.fit_transform(X_test[:, 8] )

df=pd.DataFrame(X_test)

onehotencoder_test_embarked = OneHotEncoder(categorical_features = [2])
X_test = onehotencoder_test_embarked.fit_transform(X_test).toarray()

df=pd.DataFrame(X_test)
# Avoiding the Dummy Variable Trap
X_test = X_test[:, 1:]

df=pd.DataFrame(X_test)

onehotencoder = OneHotEncoder(categorical_features = [8])
X_test = onehotencoder.fit_transform(X_test).toarray()

df=pd.DataFrame(X_test)

X_test = X_test[:, 1:]

df=pd.DataFrame(X_test)

from sklearn.ensemble import RandomForestRegressor
regressorrf = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorrf.fit(X, y)

regressorrf.score(X, y)

from sklearn.tree import DecisionTreeRegressor
regressordt = DecisionTreeRegressor(random_state = 0)
regressordt.fit(X, y)

# Predicting the Test set results
y_pred = np.round(regressordt.predict(X_test))



