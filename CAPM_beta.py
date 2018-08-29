#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:29:41 2018

@author: darshandoshi
"""

#CAPM beta
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm

 
MCD_data = pd.read_csv("/Users/darshandoshi/Desktop/Semester3/MSF/MCD.csv",sep=",")
market_data=pd.read_csv("/Users/darshandoshi/Desktop/Semester3/MSF/^GSPC.csv",sep=",")
X =market_data['Close'].pct_change(1).dropna(axis=0)
Y= MCD_data['Close'].pct_change(1).dropna(axis=0)

X1 = sm.add_constant(X)
X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=1/3,random_state=0)
model_ols=LinearRegression()
model_ols.fit(X_train,Y_train)
results_ols=model_ols.predict(X_test)
print(model_ols.coef_)


model = sm.OLS(Y, X1)
results = model.fit()
print(results_ols.summary())