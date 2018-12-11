
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import random
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
from sklearn import preprocessing

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[5]:


data = pd.read_csv('breast cancer dataset.csv')

# feature names as a list
col = data.columns

# y includes our labels and x includes our features
y= data.diagnosis                          # M or B 
a = ['Unnamed: 32','id','diagnosis']
x_train = data.drop(a,axis = 1 )              #original training data


# In[6]:


df_scaled = preprocessing.scale(x_train) #z-score
x=pd.DataFrame(df_scaled)  #array transfer into dataframe


# In[7]:


best_feature_set_1=[0, 6, 26, 10, 25, 5, 1, 16, 24]
adjust_feature_set=[4, 8, 9, 11, 14, 15, 17, 18, 19, 28, 29]
x_best_set_1=x[best_feature_set_1]


# In[10]:


# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = x.corr().abs()
corr_matrix.head()

x.T.corr()
df_pc=x.T.corr().abs()
#df_pc=df_pc[df_pc>0.8]

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
#df_pc_sum=df_pc.sum()
#df_pc_sum
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
to_drop_pd=pd.DataFrame(to_drop)
x=x.drop(columns = to_drop)
x.shape


# In[11]:


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


# In[15]:


def pearson_correlation_array(df):
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.9]
    df_pc[np.isnan(df_pc)]=0
    df_pc_sum=df_pc.sum()
    return (df_pc_sum)
max(pearson_correlation_array(x))


# In[14]:


def group_correlation_array(df):
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.9]
    df_pc[np.isnan(df_pc)]=0
    df_pc_dummy=df_pc.mask(df_pc>0,1)
    df_pc_dummy_sum=df_pc_dummy.sum()
    return (df_pc_dummy_sum)
max(group_correlation_array(x))


# In[17]:


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


# In[18]:


sensitivity_array=[]
group_sensitivity_array=[]
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
    acc_difference=abs(acc_score_array_del-acc_array[k])
    sensitivity=max(pearson_correlation_array(fea_drop_train)*acc_difference)
    sensitivity_array.append(sensitivity)
    
    group_sensitivity=max(group_correlation_array(fea_drop_train)*acc_difference)
    group_sensitivity_array.append(group_sensitivity)
print(sensitivity_array)
print(group_sensitivity_array)


# In[20]:




