#!/usr/bin/env python
# coding: utf-8

# In[17]:

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy as sp
import seaborn as sns
from sklearn import metrics


# In[2]:

#Importing Data
dataset = pd.read_csv('house_price.csv')


# In[3]:

#Creating data for model
dataset=dataset.drop(['zipcode', 'lat', 'long','id','date'], axis=1)


# In[4]:


dataset.describe()


# In[5]:

#Checking Null values
dataset. isnull(). sum()


# In[6]:

# divinding dependent and Independent variables
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[7]:

#Spliting Training and Testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state= 0)


# In[8]:


print(x)


# In[9]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, :] = sc.fit_transform(x_train[:, :])
x_test[:, :] = sc.transform(x_test[:, :])
print(x_train)
print(x_test)


# In[10]:

#Model Building
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[11]:

#Predicting result
y_pred=regressor.predict(x_test)


# In[12]:

#cheaking result with predicted data
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[13]:

#Checking R square value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[14]:

#Coeeficient values
print(regressor.intercept_)
print(regressor.coef_)


# In[18]:

#Calculating RMSE
print(metrics.mean_squared_error(y_test, y_pred))


# In[20]:

#Checking R square value
print(regressor.score(x_test,y_test))







