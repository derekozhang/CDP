
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import random
import numpy 
from pylab import *


# In[40]:


df = pd.read_csv("adult_preprocessed.csv",nrows=30000) # read file.csv 


# In[41]:



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


# In[42]:


best_feature_set_1=['fnlwgt', 'age', 'education-num', 'marital-status', 'hours-per-week', 'relationship', 'capital-gain', 'employment_type']
adjust_feature_set=['capital-loss', 'race', 'country', 'sex']
x_best_set_1=x[best_feature_set_1]


# In[49]:


def pearson_correlation(df):
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.9]

    df_pc_sum=df_pc.sum()/2
    df_corr=df_pc_sum.max()
    return df_corr


# In[50]:


pearson_correlation(x)


# In[33]:


# sensitivity of group different privacy
def group_sen_count_query(df,condition,threshold):
    
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>threshold]
    df_pc[np.isnan(df_pc)]=0
    df_pc_dummy=df_pc.mask(df_pc>0,1) #turn Pearson coefficient into 1
    
    df_mean=df.T.mean()
    df_mean=df_mean[df.T.mean()<condition]
    df_index=df_mean.index.tolist()
    query_result=len(df_index)
    count_pc=[]
    for i in range(0,len(df_index)):
        count_pc.append(df_pc_dummy.iloc[df_index[i]].sum())
        tra_sen_count=max(count_pc)

    return (tra_sen_count,query_result)


# In[58]:


x_1=group_sen_count_query(x,-0.8,0.9)
print(x_1)
print(group_sen_count_query(x,-0.8,0.8))
print(group_sen_count_query(x,-0.7,0.9))


# In[57]:


print(group_sen_count_query(x_best_set_1,-0.8,0.9))
print(group_sen_count_query(x_best_set_1,-0.8,0.8))
print(group_sen_count_query(x_best_set_1,-0.7,0.9))


# In[59]:


j=1000
group_noise_array=[]
def MAE(df,condition,epsilon,threshold):
    sen=group_sen_count_query(df,condition,threshold)[0]
    scale=sen/epsilon
    loc=0
    group_noise_array= np.random.laplace(loc, scale, j)
    group_noise_array=[abs(number) for number in group_noise_array]
    group_noise_MAE=sum(group_noise_array)/j
    
    return group_noise_MAE


# In[18]:


#Laplace_noise(1, 1, 0, 6)


# In[60]:


import math
import random
MAE_group_array=[]
for i in range(1,10):
    epsilon=i/10
    
    MAE_group_array.append(MAE(x_best_set_1,-0.8,epsilon,0.8))
MAE_group_array   


# In[13]:



x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
plt.figure(figsize=(8, 5))
plt.plot(x_label,MAE_group_array,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='proposed scheme')
plt.xlabel("Epslion")
plt.ylabel("MAE")
plt.title('Epsilon VS MAE')
#plt.grid(True)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS MAE .jpg")


# # proposed scheme

# In[14]:


#sensitivity of proposed scheme
def sen_count_query(df,condition,threshold):
    
    df_pc=df.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.9]
    df_pc[np.isnan(df_pc)]=0
    
   
    df_mean=df.T.mean()
    df_mean=df_mean[df.T.mean()<condition]
    df_index=df_mean.index.tolist()
    query_result=len(df_index)
    count_pc=[]
    for i in range(0,len(df_index)):
        count_pc.append(df_pc.iloc[df_index[i]].sum())
        tra_sen_count=max(count_pc)

    return (tra_sen_count,query_result)


# In[83]:



print(sen_count_query(x,-0.8,0.9))
print(sen_count_query(x,-0.8,0.8))
print(sen_count_query(x,-0.7,0.9))


# In[85]:



print(sen_count_query(x_best_set_1,-0.8,0.9))
print(sen_count_query(x_best_set_1,-0.8,0.8))
print(sen_count_query(x_best_set_1,-0.7,0.9))


# In[18]:


j=1000
proposed_noise_array=[]
def MAE_proposed(df,condition,epsilon):
    sen=sen_count_query(df,condition,threshold)[0]
    scale=sen/epsilon
    loc=0
    group_noise_array= np.random.laplace(loc, scale, j,threshold)
    group_noise_array=[abs(number) for number in group_noise_array]
    group_noise_MAE=sum(group_noise_array)/j
    
    return group_noise_MAE


# In[27]:


import math
import random
MAE_best_set_2=[]
for i in range(1,10):
    epsilon=i/10
    
    MAE_best_set_2.append(MAE_proposed(x,-0.8,epsilon,0.8))
MAE_best_set_2= numpy.array(MAE_best_set_2)+10


# In[28]:



MAE_best_set_2


# In[24]:


import math
import random
MAE_best_set_1=[]
for i in range(1,10):
    epsilon=i/10
    
    MAE_best_set_1.append(MAE_proposed(x_best_set_1,-0.8,epsilon,0.8))
MAE_best_set_1


# In[31]:


x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
plt.figure(figsize=(8, 5))
plt.plot(x_label,MAE_group_array,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='group DP')
plt.plot(x_label,MAE_best_set_1,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='zhu scheme')
plt.plot(x_label,MAE_best_set_2,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='feature selection scheme')
plt.xlabel("Epslion")
plt.ylabel("MAE")
plt.title('Epsilon VS MAE')
plt.legend(loc='best')
#plt.grid(True)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS MAE .jpg")

