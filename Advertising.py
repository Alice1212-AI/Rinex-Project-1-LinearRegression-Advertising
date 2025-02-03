import pandas as pd
data = pd.read_csv("Advertising.csv",index_col=0)
data.head()
data.tail()
data.shape
import seaborn as sns
%matplotlib inline
sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=7,aspect=0.7,kind='reg')
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
X=data[['TV','Radio','Newspaper']]
X.head()
print(type(X))
print(X.shape)
y=data['Sales']
y=data.Sales
y.head()
print(type(y))
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
list(zip(feature_cols,linreg.coef_))
y_pred=linreg.predict(X_test)
true=[100,50,30,20]
pred=[90,50,50,30]
print((10+0+20+10)/4.)
from sklearn import metrics
print(metrics.mean_absolute_error(true,pred))
print((10**2+0**2+20**2+10**2)/4)
print(metrics.mean_squared_error(true,pred))
import numpy as np
print(np.sqrt((10**2+0**2+20**2+10**2)/4.))
print(np.sqrt(metrics.mean_squared_error(true,pred)))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
feature_cols=['TV','Radio']
X=data[feature_cols]
y=data.Sales
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))