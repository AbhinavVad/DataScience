#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
data1 = pd.DataFrame([[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
					 [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]])

data1 = data1.T #applying transpose

#splitting data into x,y
x = data1.loc[:,0] #Hours
y = data1.loc[:,1] #Scores

#plt.scatter(x,y) #linear relation present
#plt.xlabel("No. of Hours")
#plt.ylabel("Scores")
print(np.corrcoef(x,y)) # Strong positive correlation - 0.9762
#print(data1.describe())
#data1.boxplot() #checking for outliers - none

#applying linReg
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
x = x.values.reshape(-1,1)
reg.fit(x, y)

#Prediction for 9.25
print(reg.predict([[9.25]])) #92.90985477
