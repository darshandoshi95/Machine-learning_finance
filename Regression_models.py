#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:53:09 2018

@author: darshandoshi
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
'''
st=pd.read_csv("/Users/darshandoshi/Desktop/Semester3/MSF/Stock performance.csv",sep=",")

#Create Independent variables
X=st[['Large BP','Large ROE','Large SP', 'Large Return Rate', 'Large Market Value','Small Systematic Risk']]

#Create target variables
Y1=st['Annual Return']
Y2=st['Annual Return N']

#Linear Regression
#target variable- Annual return
X_train,X_test,Y1_train,Y1_test=train_test_split(X,Y1,test_size=0.2,random_state=0)
Linear_model=LinearRegression()
Linear_model.fit(X,Y1)
Y1_pred=Linear_model.predict(X_test)
print('R squared error=%f' %(metrics.explained_variance_score(Y1_test,Y1_pred)))
print('mean squared error=%f' %(metrics.mean_squared_error(Y1_test,Y1_pred)))
print('mean absolute error=%f' %(metrics.mean_absolute_error(Y1_test,Y1_pred)))
print('Coefficients=',Linear_model.coef_)
print('Intercept=',Linear_model.intercept_)

#target variable- Annual Return N
X_train,X_test,Y2_train,Y2_test=train_test_split(X,Y2,test_size=0.2,random_state=0)
Linear_model=LinearRegression()
Linear_model.fit(X,Y1)
Y2_pred=Linear_model.predict(X_test)
print('R squared error=%f' %(metrics.explained_variance_score(Y2_test,Y2_pred)))
print('mean squared error=%f' %(metrics.mean_squared_error(Y2_test,Y2_pred)))
print('mean absolute error=%f' %(metrics.mean_absolute_error(Y2_test,Y2_pred)))
print('Coefficients=',Linear_model.coef_)
print('Intercept=',Linear_model.intercept_)

#----- ridge regression -------
from sklearn.linear_model import Ridge
#target variable-  Annual return
alpha=0.005
Ridge_model=Ridge(alpha)
Ridge_model.fit(X_train,Y1_train)
Y1_pred=Ridge_model.predict(X_test)
print('R squared error=%f' %(metrics.explained_variance_score(Y1_test,Y1_pred)))
print('mean squared error=%f' %(metrics.mean_squared_error(Y1_test,Y1_pred)))
print('mean absolute error=%f' %(metrics.mean_absolute_error(Y1_test,Y1_pred)))
print('Coefficients=',Ridge_model.coef_)
print('Intercept=',Ridge_model.intercept_)


#target variable- Annual Return N
alpha=0.1
Ridge_model=Ridge(alpha)
Ridge_model.fit(X_train,Y2_train)
Y2_pred=Ridge_model.predict(X_test)
print('R squared error=%f' %(metrics.explained_variance_score(Y2_test,Y2_pred)))
print('mean squared error=%f' %(metrics.mean_squared_error(Y2_test,Y2_pred)))
print('mean absolute error=%f' %(metrics.mean_absolute_error(Y2_test,Y2_pred)))
print('Coefficients=',Ridge_model.coef_)
print('Intercept=',Ridge_model.intercept_)
'''
def collect_data():
    st=pd.read_csv("/Users/darshandoshi/Desktop/Semester3/MSF/Stock performance.csv",sep=",")
    print("1")
    set_variables(st)

def set_variables(st):
    X=st[['Large BP','Large ROE','Large SP', 'Large Return Rate', 'Large Market Value','Small Systematic Risk']]
    #Create target variables
    Y1=st['Annual Return']
    Y2=st['Annual Return N']
    #X_train,X_test,Y1_train,Y1_test=train_test_split(X,Y1,test_size=0.2,random_state=0)
    regression(X,Y1,Y2,2)

def regression(X,Y1,Y2,temp):
    
    results_1=pd.DataFrame()
    results_2=pd.DataFrame(columns=['Method','R sqaured error','mean squared error','Coefficients','Intercept'])

    method=dict(linear_model=LinearRegression(),ridge_model=Ridge(alpha=0.1),Lasso_model=Lasso(alpha=1,max_iter=50000))
    while temp!=0:
        print(temp)
        
        if temp==2:
            print("----------- Annual Return -------------")
            X_train,X_test,Y1_train,Y1_test=train_test_split(X,Y1,test_size=0.2,random_state=0)
            for name, model in method.items():
                results_1['Method']=name
                model.fit(X,Y1)
                Y1_pred=model.predict(X_test)
                '''
                results_1['R squared error']= (metrics.explained_variance_score(Y1_test,Y1_pred))
                results_1['mean squared error'] =(metrics.mean_squared_error(Y1_test,Y1_pred))
                results_1['mean absolute error']= (metrics.mean_absolute_error(Y1_test,Y1_pred))
                results_1['Coefficients']=model.coef_
                results_1 ['Intercept']=model.intercept_
                
                '''
                print('\n'+name)
                print('R squared error=%f' %(metrics.explained_variance_score(Y1_test,Y1_pred)))
                print('mean squared error=%f' %(metrics.mean_squared_error(Y1_test,Y1_pred)))
                print('mean absolute error=%f' %(metrics.mean_absolute_error(Y1_test,Y1_pred)))
                print('Coefficients=',model.coef_)
                print('Intercept=',model.intercept_)
                
                results_1=results_1.append({'Model':name,'R squared error':metrics.explained_variance_score(Y1_test,Y1_pred),
                'mean squared error':(metrics.mean_squared_error(Y1_test,Y1_pred)),
                'mean absolute error':(metrics.mean_absolute_error(Y1_test,Y1_pred)),
                'Coefficients':model.coef_,
                'Intercept':model.intercept_},ignore_index=True)
                #print(results_1)
            temp=temp -1
        elif temp==1:
            print("----------- Annual Return  N-------------")
            X_train,X_test,Y2_train,Y2_test=train_test_split(X,Y2,test_size=0.2,random_state=0)
            for name, model in method.items():
                results_2['Method']=name
                model.fit(X,Y2)
                Y2_pred=model.predict(X_test)
                
                print('\n'+name)
                print('R squared error=%f' %(metrics.explained_variance_score(Y2_test,Y2_pred)))
                print('mean squared error=%f' %(metrics.mean_squared_error(Y2_test,Y2_pred)))
                print('mean absolute error=%f' %(metrics.mean_absolute_error(Y2_test,Y2_pred)))
                print('Coefficients=',model.coef_)
                print('Intercept=',model.intercept_)
                results_2=results_2.append({'Model':name,'R squared error':metrics.explained_variance_score(Y2_test,Y2_pred),
                'mean squared error':(metrics.mean_squared_error(Y2_test,Y2_pred)),
                'mean absolute error':(metrics.mean_absolute_error(Y2_test,Y2_pred)),
                'Coefficients':model.coef_,
                'Intercept':model.intercept_},ignore_index=True)
                
            temp = temp-1
            
  
collect_data()




        
        
      

    


    
    
    