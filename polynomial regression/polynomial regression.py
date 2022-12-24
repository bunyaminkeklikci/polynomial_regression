import pandas as pd

data=pd.read_csv("C:/Users/kekli/Desktop/ev.csv")
veri=data.copy()

#print(veri.shape)
#print(veri.info())

veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"],axis=1,inplace=True)
#print(veri)

veri=veri.rename(columns={"X2 house age":"Ev Yası",
"X3 distance to the nearest MRT station":"Metro Uzaklık",
"X4 number of convenience stores":"Market Sayısı",
"Y house price of unit area":"Evin Fiyatı"})

#print(veri)

#print(veri.isnull().sum()) eksik gözlem var mı

import matplotlib.pyplot as plt
import seaborn as sns

#sns.pairplot(veri) 
#plt.show()

y= veri["Evin Fiyatı"]
X=veri.drop(columns="Evin Fiyatı",axis = 1)
#print(X)

#egitim ve test verileri ayırma
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt

#bağımsız değişkenleri polinomsal değere atadık
pol = PolynomialFeatures(degree=2)
X_pol = pol.fit_transform(X)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin2=lr.predict(X_test)

Rr2=mt.r2_score(y_test,tahmin2)
RMse =mt.mean_squared_error(y_test,tahmin2)
 

#parçalama işlemi
X_train,X_test,y_train,y_test=train_test_split(X_pol,y,test_size=0.2,random_state=42)
# model kurulumu
pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train) 
tahmin = pol_reg.predict(X_test)

r2=mt.r2_score(y_test,tahmin)
Mse =mt.mean_squared_error(y_test,tahmin)

print("R2:{} MSE: {}".format(r2,Mse))
print("RR2:{} RMSE: {}".format(Rr2,RMse))








