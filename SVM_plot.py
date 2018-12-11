
# coding: utf-8

# In[2]:


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


# In[3]:


df = pd.read_csv("adult_preprocessed.csv",nrows=5000) # read file.csv from sklearn.preprocessing import StandardScaler
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


# In[117]:


best_feature_set_1=['fnlwgt', 'age', 'education-num', 'marital-status', 'hours-per-week', 'relationship', 'capital-gain', 'employment_type']
adjust_feature_set=['capital-loss', 'race', 'country', 'sex']
x_best_set_1=x[best_feature_set_1]


# In[65]:


x.shape


# In[4]:


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


# In[48]:


from sklearn.svm import SVC
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=SVC(kernel = 'linear')

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_SVC = accuracy_score(y_val,prediction)
print(acc_score_SVC)
print('w = ',model.coef_)
print('b = ',model.intercept_)
print('Indices of support vectors = ', model.support_)
print('Support vectors = ', model.support_vectors_)
print('Number of support vectors for each class = ', model.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(model.dual_coef_))


# In[107]:


proposed_sensitivity=0.84
zhu_sensitivity=1.37
group_sensitivity=1.45


# In[92]:


svm_coef=[-0.24617463, -0.05269883, -0.74548103,  0.80338565,  0.12730395,  0.08338186 ,
          0.13052734, -0.38721462, -0.15864332, -0.2266802,  -0.05608071,  0.03691698]


# In[93]:


svm_coefficience=[-0.24617463, -0.05269883, -0.74548103,  0.80338565,  0.12730395,  0.08338186 ,
          0.13052734, -0.38721462, -0.15864332, -0.2266802,  -0.05608071,  0.03691698, 1.3588728]
svm_b=1.3588728


# In[134]:


#svm training results without noise
def svm_result(df,w,b):
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


# In[67]:





# In[118]:


from sklearn.svm import SVC
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(x_best_set_1,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=SVC(kernel = 'linear')

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_SVC = accuracy_score(y_val,prediction)
print(acc_score_SVC)
print('w = ',model.coef_)
print('b = ',model.intercept_)
print('Indices of support vectors = ', model.support_)
print('Support vectors = ', model.support_vectors_)
print('Number of support vectors for each class = ', model.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(model.dual_coef_))


# In[148]:


svm_coef_2=[-0.03520508, -0.23478565, -0.82818484,  0.76741234, -0.21707211,  0.13569639,
  -0.38895975,  0.05097874]
svm_b_2=1.35050051
print(svm_result(x,svm_coef,svm_b))
print(svm_result(x_best_set_1,svm_coef_2,svm_b_2))  # accuracy without noise scheme
y_svm_no_noise=[0.8236, 0.8236, 0.8236, 0.8236, 0.8236, 0.8236, 0.8236, 0.8236, 0.8236]


# In[172]:


def svm_noise_result(df,w,b,sen):
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


# In[173]:


print(svm_noise_result(x,svm_coef,svm_b,0.84))
print(svm_noise_result(x_best_set_1,svm_coef_2,svm_b_2,1.45))


# In[174]:


def svm_noise_result_mean(df,w,b,sen):
    array=[]
    for i in range(0,300):
        array.append(svm_noise_result(x,svm_coef,svm_b,sen))
    result=sum(array)/len(array)
    
    return (result)


# In[175]:


print(svm_noise_result_mean(x,svm_coef,svm_b,0.84))   #proposedd scheme
print(svm_noise_result_mean(x_best_set_1,svm_coef_2,svm_b_2,1.45))     #group scheme
print(svm_noise_result_mean(x_best_set_1,svm_coef_2,svm_b_2,1.37)) #zhu scheme


# In[177]:


def plot_svm_noise_result(df,w,b,sen):
    array=[]
    for i in range(1,10):
        epsilon=i/10 
        array.append(svm_noise_result_mean(df,w,b,sen/epsilon))
    return (array)
        


# In[178]:


y_svm_proposed=plot_svm_noise_result(x,svm_coef,svm_b,0.84)
y_svm_group=plot_svm_noise_result(x_best_set_1,svm_coef_2,svm_b_2,1.45)
y_svm_zhu=plot_svm_noise_result(x_best_set_1,svm_coef_2,svm_b_2,1.37)


# In[ ]:


x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
plt.figure(figsize=(8, 5))
plt.plot(x_label,y_svm_proposed,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='proposed scheme')
plt.plot(x_label,y_svm_group,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='group scheme')
plt.plot(x_label,y_svm_zhu,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='zhu scheme')
plt.plot(x_label,y_svm_no_noise,linewidth=2.0,color='pink',markerfacecolor='pink',marker='o',label='No noise scheme')
plt.xlabel("Epslion")
plt.ylabel("Accuracy")
plt.title('Epsilon VS Accuracy')
plt.legend(loc='best')
#plt.grid(True)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS Accuracy SVM .png")


# In[195]:


# import numpy as np
# from sklearn.linear_model import LinearRegression
# reg = LinearRegression().fit(x, y)
# reg.score(x, y)
# reg.coef_
# reg.intercept_

