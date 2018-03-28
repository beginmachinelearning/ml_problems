empty_columns=check_empty_value_columns(dataset)

y_train=np.asarray(dataset.iloc[:, 1].values)
X_train=np.asarray(dataset.iloc[:, :].values)
X_train=np.delete(X_train, 1, axis=1)


age_mean=pd.DataFrame(X_train[:, 4]).mean()
K=pd.DataFrame(X_train[:, 4]).fillna(age_mean[0]).iloc[:, :]
X_train[:, 4]= K.ravel()

X= pd.DataFrame(X_train)
T=np.equal(X_train,'Nan')

print((dataset[[1,2,3,4,5]] == 0).sum())

print(maxim())