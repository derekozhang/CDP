
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import random
from pylab import *

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[85]:


df = pd.read_csv("adult_preprocessed.csv",nrows=5000) # read file.csv 


# In[33]:



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


# In[79]:


best_feature_set_1=['fnlwgt', 'age', 'education-num', 'marital-status', 'hours-per-week', 'relationship', 'capital-gain', 'employment_type']
adjust_feature_set=['capital-loss', 'race', 'country', 'sex']
x_best_set_1=x[best_feature_set_1]


# In[84]:


#correlated degree matrix for proposed scheme
def pearson_correlation_array(df):
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.8]
    df_pc[np.isnan(df_pc)]=0
    df_pc_sum=df_pc.sum()/2
    return (df_pc_sum)

#print(pearson_correlation_array(x))
#print(pearson_correlation_array(x_best_set_1))


# In[35]:


#correlated degree matrix for group DP
df_pc=x.T.corr()
df_pc=df_pc.replace(1,0)
df_pc=abs(df_pc)
df_pc=df_pc[df_pc>0.8]
df_pc[np.isnan(df_pc)]=0
df_pc_dummy=df_pc.mask(df_pc>0,1)
df_pc_dummy_sum=df_pc_dummy.sum()/2


# In[52]:


# maka an order for feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
fea_im_seq=[] 

for i in range(0,100):
    model= RandomForestClassifier()      
    model=model.fit(x,y)              #dont miss the later model.fit
    feature_importances = np.zeros(x.shape[1])
    feature_importances = model.feature_importances_
    fea_im_seq.append(feature_importances)

fea_im_seq=pd.DataFrame(fea_im_seq)
fea_im_seq=fea_im_seq.mean() #feature importance average
fea_im_seq = pd.DataFrame({'feature': x.columns, 'importance': fea_im_seq}).sort_values('importance', ascending = False)
fea_im_seq


# In[74]:


# the begining accuracy for dataset x 
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
acc_score_array=[]
split_size=0.3
acc_array=[]
for j in range(0,len(adjust_feature_set)):
        
    if j==0:
        fea_drop=fea_im_seq
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]] 
    else:
        fea_drop=fea_im_seq[:-j]    #drop last features
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset

    for i in range(0,30):

        #Creation of Train and Test dataset
        X_train, X_test, y_train, y_test = train_test_split( fea_drop_train,y,test_size=split_size,random_state=22)

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
    acc_array.append(acc_score)
print(acc_array)


# In[78]:


#sensitivity of proposed scheme and corresbonding group scheme
acc_score_array_del=[]
for i in range(0,x.shape[0]):
    x_del=x.drop(i)
    y_del=y.drop(i)
    acc_score_array=[]
    split_size=0.3
    
    for j in range(0,5):

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
    
acc_difference=acc_score_array_del-acc_score

sensitivity=max(pearson_correlation_array(x)*acc_array[0])
print(sensitivity)

group_sensitivity=max(df_pc_dummy_sum*acc_array[0])
print(group_sensitivity)


# In[40]:





# In[80]:


sensitivity_array=[]
for k in range(0,len(adjust_feature_set)):
    
    if k==0:
        fea_drop=fea_im_seq
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]] 
    else:
        fea_drop=fea_im_seq[:-k]    #drop last features
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset


    acc_score_array_del=[]
    for i in range(0,x_best_set_1.shape[0]):
        x_del=fea_drop_train.drop(i)
        y_del=y.drop(i)
        acc_score_array=[]
        split_size=0.3

        for j in range(0,5):

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
    acc_difference=acc_score_array_del-acc_array[l]
    sensitivity=max(pearson_correlation_array(fea_drop_train)*acc_difference)
    sensitivity_array.append(sensitivity)
print(sensitivity_array)


# In[65]:


sensitivity_array


# In[76]:


# the sensitivity of best feature set I (less features)

acc_score_array_del=[]
for i in range(0,x.shape[0]):
    x_del=x_best_set_1.drop(i)
    y_del=y.drop(i)
    acc_score_array=[]
    split_size=0.3
    
    for j in range(0,1):

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


acc_difference=acc_score_array_del-acc_score

sensitivity=max(pearson_correlation_array(x_best_set_1)*acc_difference)
print(sensitivity)


# In[ ]:





# In[ ]:




