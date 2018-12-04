
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


df = pd.read_csv("adult_preprocessed",nrows=3000) # read file.csv 

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


# In[39]:


x


# In[15]:


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


# In[5]:


# pearson correlation coefficient
def pearson_correlation(df):
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.9]

    df_pc_sum=df_pc.sum()/2
    df_corr=df_pc_sum.max()
    return df_corr


# In[38]:


fea_ac_array=[]
fea_ac_matrix=[]
split_size=0.3     

for j in range(0,3000):
    fea_ac_array=[]
    for i in range(0,x.shape[1]):
        if i==0:
            fea_drop=fea_im_ave    #drop last features
            fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset   
        else:
            fea_drop=fea_im_ave[:-i]    #drop last features
            fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset

        X_train, X_test, y_train, y_test = train_test_split(fea_drop_train,y,test_size=split_size,random_state=22)
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)
        model=RandomForestClassifier()
        kfold = model_selection.KFold(n_splits=5,random_state=7)
        cv_result = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
        score=model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        fea_ac= accuracy_score(y_val,prediction)
        fea_ac_array.append(fea_ac)
    fea_ac_matrix.append(fea_ac_array)
    
fea_ac_matrix=pd.DataFrame(fea_ac_matrix)
fea_ac_array=fea_ac_matrix.mean() 
fea_ac_array


# In[16]:


fea_sum_array=[]
for i in range(0,x.shape[1]):
    
    fea_drop=fea_im_ave[:-i]    #drop last features
    fea_sum_array.append(fea_drop.sum())
    
fea_sum_array=pd.DataFrame(fea_sum_array)
fea_sum_array=fea_sum_array.replace(0,1)
fea_sum_array=fea_sum_array.iloc[:,1]

print(fea_sum_array)


# In[17]:


plt.figure(figsize=(8, 5))
plt.plot(fea_sum_array,fea_ac_array,linewidth=2.0,color='red',markerfacecolor='blue',marker='o')
plt.xlabel("Feature importance")
plt.ylabel("Accuracy")
plt.title('Accuracy VS Feature importance in Adult dataset')
#plt.grid(True)
xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Accuracy VS Feature importance.jpg")


# In[37]:


# find best feature set I and adjusted feature set
fea_ac_list=list(fea_ac_array)
fea_ac_position=fea_ac_list.index(max(fea_ac_list))
best_feature_set_1=fea_im_seq['feature'].tolist()[:(x.shape[1]-fea_ac_position)]
adjust_feature_set=list(set(x)-set(best_feature_set_1))
print(best_feature_set_1)
print(adjust_feature_set)

