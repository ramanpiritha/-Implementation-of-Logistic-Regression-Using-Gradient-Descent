# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: G.HINDHU
RegisterNumber:  212223230079
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

## Output:
## Dataset

![image](https://github.com/user-attachments/assets/99b29920-0894-496f-bf26-54fb8f273971)


## Dataset.dtypes
![image](https://github.com/user-attachments/assets/a301c034-3151-413c-b8a3-773cf0ebcdaf)

## Dataset
![image](https://github.com/user-attachments/assets/6952986b-ae30-4f05-98cb-040ed60b5a66)


## Y
![image](https://github.com/user-attachments/assets/103b4ecd-a23c-4255-8e08-b8682722f085)

## Accuracy
![image](https://github.com/user-attachments/assets/77dd10dc-c7d9-4a26-94ec-e16e0c2fb0b3)

## Y_pred
![image](https://github.com/user-attachments/assets/f89ec6c8-1ea3-4e37-8932-ecda2d57f9b6)

## Y
![image](https://github.com/user-attachments/assets/2d8606df-5b19-4860-985d-0e03caf1f92d)

## Y_prednew
.![image](https://github.com/user-attachments/assets/c88c1298-57c5-4535-9411-2b3a788e85bc)

## Y_prednew
![image](https://github.com/user-attachments/assets/666a54ec-34bf-4c3b-bf93-a90789eb10f7)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

