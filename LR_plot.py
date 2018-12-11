
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import random
from pylab import *

import sys
import warnings
from sklearn.model_selection import train_test_split
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[20]:


df = pd.read_csv("adult_preprocessed.csv",nrows=3000) # read file.csv from sklearn.preprocessing import StandardScaler
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


# In[21]:


best_feature_set_1=['fnlwgt', 'age', 'education-num', 'marital-status', 'hours-per-week', 'relationship', 'capital-gain', 'employment_type']
adjust_feature_set=['capital-loss', 'race', 'country', 'sex']
x_best_set_1=x[best_feature_set_1]


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
split_size=0.3

X_train, X_test, y_train, y_test = train_test_split( x,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=LogisticRegression()

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_LR = accuracy_score(y_val,prediction)
print(acc_score_LR)
print(('w = ',model.coef_))
print('b = ',model.intercept_)


# In[23]:


proposed_sensitivity=0.84
zhu_sensitivity=1.37
group_sensitivity=1.45


# In[24]:


LR_coef=[-0.40290828,
         -0.03813246, -0.82944237,  1.16708601,  0.2401368 ,
         0.11110091,  0.0411238 , -0.42984001, -0.22745116, -0.32942755,
        -0.06856892,  0.08034085]
LR_b=1.83172074


# In[25]:


def LR_result(df,w,b):
    predicted_result=[]
    for i in range(0,df.shape[0]):
        
        a=np.dot(df.iloc[i].tolist(),w)+b
        predicted_result.append(a)
        
    predicted_result=pd.DataFrame(predicted_result)
    predicted_result=predicted_result.mask( predicted_result< 0, 0)
    predicted_result=predicted_result.mask( predicted_result>0, 1)
    diff_result=(predicted_result[0].tolist()-y).tolist()
    acc=1-(df.shape[0]-diff_result.count(0))/df.shape[0]
    return (acc)


# In[26]:


LR_result(x,LR_coef,LR_b)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
split_size=0.3

X_train, X_test, y_train, y_test = train_test_split( x_best_set_1,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=LogisticRegression()

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_LR = accuracy_score(y_val,prediction)
print(acc_score_LR)
print(('w = ',model.coef_))
print('b = ',model.intercept_)


# In[37]:


LR_coef_2=[-0.01201438, -0.40781279, -0.84389346,  1.15111337, -0.3458217 ,
         0.25612206, -0.4164229 ,  0.0788421]
LR_b_2=1.82139654
LR_result(x_best_set_1,LR_coef_2,LR_b_2)
y_LR_no_noise=[0.8236,0.8236,0.8236,0.8236,0.8236,0.8236,0.8236,0.8236,0.8236]


# In[29]:


def LR_noise_result(df,w,b,sen):
    loc=0
    scale=sen
    noise_1= np.random.laplace(loc, scale, df.shape[1])

    noise_2=np.random.laplace(loc, scale, 1)

    svm_coef_noise=noise_1+w
    svm_b_noise=noise_2+b

    predicted_result_noise=[]
    for i in range(0,df.shape[0]):
        predicted_result_noise.append(np.dot(df.iloc[i].tolist(),svm_coef_noise)+svm_b_noise)
    
    

    predicted_result_noise=pd.DataFrame(predicted_result_noise)


    predicted_result_noise=predicted_result_noise.mask( predicted_result_noise< 0, 0)

    predicted_result_noise=predicted_result_noise.mask( predicted_result_noise>0, 1)

    diff_result_noise=(predicted_result_noise[0].tolist()-y).tolist()
    acc=1-(df.shape[0]-diff_result_noise.count(0))/df.shape[0]
    return (acc)


# In[36]:


print(LR_noise_result(x,LR_coef,LR_b,0.84))
print(LR_noise_result(x_best_set_1,LR_coef_2,LR_b_2,1.45))  #baseline of no noise scheme


# In[ ]:


def LR_noise_result_mean(df,w,b,sen):
    array=[]
    for i in range(0,30):
        array.append(LR_noise_result(df,w,b,sen))
    result=sum(array)/len(array)
    
    return (result)


# In[ ]:


print(LR_noise_result_mean(x,LR_coef,LR_b,0.84))   #proposedd scheme
print(LR_noise_result_mean(x_best_set_1,LR_coef_2,LR_b_2,1.45))     #group scheme
print(LR_noise_result_mean(x_best_set_1,LR_coef_2,LR_b_2,1.37)) #zhu scheme


# In[ ]:


def plot_LR_noise_result(df,w,b,sen):
    array=[]
    for i in range(1,10):
        epsilon=i/10 
        array.append(LR_noise_result_mean(df,w,b,sen/epsilon))
    return (array)


# In[ ]:


y_LR_proposed=plot_LR_noise_result(x,LR_coef,LR_b,0.84)
y_LR_group=plot_LR_noise_result(x_best_set_1,LR_coef_2,LR_b_2,1.45)
y_LR_zhu=plot_LR_noise_result(x_best_set_1,LR_coef_2,LR_b_2,1.37)


# In[ ]:


x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
plt.figure(figsize=(8, 5))
plt.plot(x_label,y_LR_proposed,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='proposed scheme')
plt.plot(x_label,y_LR_group,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='group scheme')
plt.plot(x_label,y_LR_zhu,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='zhu scheme')
plt.plot(x_label,y_LR_no_noise,linewidth=2.0,color='pink',markerfacecolor='pink',marker='o',label='No noise scheme')
plt.xlabel("Epslion")
plt.ylabel("Accuracy")
plt.title('Epsilon VS Accuracy')
plt.legend(loc='best')
#plt.grid(True)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS Accuracy LR .png")

