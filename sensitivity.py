
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import random

from pylab import *


# In[2]:


df = pd.read_csv("adult_preprocessed.csv",nrows=3000) # read file.csv 


# In[3]:



from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
from sklearn import preprocessing

#without normolization, the pearson coefficient will be very large.
df_scaled = preprocessing.scale(df) #z-score
df=pd.DataFrame(df_scaled)  #array transfer into dataframe

df=df.drop( [0],axis='columns')
df.columns=['age',
'fnlwgt',
'education-num',
'marital-status',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'country',
'salary',
'employment_type']

df['salary']=df['salary']>0
for salary in df.columns:
    if df['salary'].dtype==bool:
        df['salary']=df['salary'].astype('int')
        
x= df.drop(['salary'],axis=1)
y=df['salary']


# In[7]:


#correlated degree matrix
df_pc=x.T.corr()
df_pc=df_pc.replace(1,0)
df_pc=abs(df_pc)
df_pc=df_pc[df_pc>0.8]
df_pc[np.isnan(df_pc)]=0


# In[22]:


# the begining accuracy for dataset x 
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
acc_score_array=[]
split_size=0.3
for i in range(0,10):

    #Creation of Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=split_size,random_state=22)

    #Creation of Train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

    model=RandomForestClassifier()
    kfold = model_selection.KFold(n_splits=5,random_state=7)
    #cv_result = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    score=model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    acc_score = accuracy_score(y_val,prediction)  
    acc_score_array.append(acc_score)

acc_score=sum(acc_score_array)/len(acc_score_array)
print(acc_score)


# In[29]:


acc_score_array_del=[]
for i in range(0,x.shape[0]):
    x_del=x.drop(i)
    y_del=y.drop(i)
    acc_score_array=[]
    split_size=0.3
    
    for j in range(0,10):

        #Creation of Train and Test dataset
        X_train, X_test, y_train, y_test = train_test_split(x_del,y_del,test_size=split_size,random_state=22)

        #Creation of Train and validation dataset
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

        model=RandomForestClassifier()
        kfold = model_selection.KFold(n_splits=5,random_state=7)
        #cv_result = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
        score=model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        acc_score = accuracy_score(y_val,prediction)  
        acc_score_array.append(acc_score)

    acc_score=sum(acc_score_array)/len(acc_score_array)
    acc_score_array_del.append(acc_score)
    


# In[39]:


acc_difference=acc_score_array_del-acc_score
df_pc_sum=df_pc.sum()/2
sensitivity=max(df_pc_sum*acc_difference)
print(sensitivity)

