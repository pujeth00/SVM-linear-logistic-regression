# perform linear regression without regularization on the california housing dataset

# import the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv('housing.csv')
df.drop('ocean_proximity',axis=1,inplace=True)

df.head()

#df.info()

#correlation matrix

corr_matrix=df.corr()

#print(corr_matrix)

data=np.array(df[0:100])
X=data[:,:-1]
Y=data[:,-1]

heading=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.scatter(X[:,i],Y,c='g',marker=".",alpha=0.5)
    plt.xlabel(heading[i])
    plt.ylabel('output')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()
split=int(0.7*X.shape[0])
print(split)


X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)

print('Coefficients: \n', reg.coef_)
print('intercept: \n', reg.intercept_)
print('Mean squared error: %.2f'% np.mean((reg.predict(X_test)-Y_test)**2))
print('reg.score: %.2f'% reg.score(X_test,Y_test))

from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(LinearRegression(),X_train,Y_train,cv=10))

plt.scatter(Y_test,Y_pred,c='g',marker=".",alpha=0.5)
plt.show()



