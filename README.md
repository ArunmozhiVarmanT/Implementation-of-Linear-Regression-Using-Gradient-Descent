# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.


2.Write a function computeCost to generate the cost function.


3.Perform iterations og gradient steps with learning rate.


4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
Name : Arunmozhi Varman T
RegisterNumber: 212223230022


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

## data information
<img width="558" height="222" alt="a1" src="https://github.com/user-attachments/assets/52538978-e360-4ce4-b6f1-b672d0687035" />

## Value of x
<img width="225" height="713" alt="a2" src="https://github.com/user-attachments/assets/964a451a-64b4-49a7-9ae9-d8d320d7c026" />

## Value of X1_scaled
<img width="343" height="707" alt="a3" src="https://github.com/user-attachments/assets/9e54daa8-a9ec-477d-8e77-49c1216ded09" />

## predicted value
<img width="247" height="46" alt="a4" src="https://github.com/user-attachments/assets/3d8cc7f7-018d-4db8-9720-09c8cfdcab85" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
