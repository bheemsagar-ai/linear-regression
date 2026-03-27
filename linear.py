import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('height.csv')
print(df.head())
plt.scatter(df['Weight'],df['Height'])
plt.xlabel(df['Weight'])
plt.ylabel(df['Height'])
plt.show()
## divide the data into independent and dependent
x=df[['Weight']]
y=df['Height']
##split train data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
##standardize the train data 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
plt.scatter(x_train,y_train)
plt.show()
## linear regression model training
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("the slope of weight",regressor.coef_)
print("intercept:",regressor.intercept_)
plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),'r')
plt.show()
y_pred_test=regressor.predict(x_test)
y_pred_test,y_test
plt.scatter(x_test,y_test)
plt.plot(x_test,regressor.predict(x_test),'r')
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred_test)
print("score",score)
adjusted = 1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adjusted R2:", adjusted)

