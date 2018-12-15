
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


best_feature_set_1=['fnlwgt', 'age', 'education-num', 'marital-status', 'hours-per-week', 'relationship', 'capital-gain', 'employment_type']
adjust_feature_set=['capital-loss', 'race', 'country', 'sex']
x_best_set_1=x[best_feature_set_1]


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
split_size=0.3

#calculate the w and b in the logistic regression
def LR_coefficient(df):

    X_train, X_test, y_train, y_test = train_test_split( df,y,test_size=split_size,random_state=22)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

    model=LogisticRegression()

    kfold = model_selection.KFold(n_splits=5,random_state=7)
    score=model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    acc_score_LR = accuracy_score(y_val,prediction)
    #print(acc_score_LR)
    LR_coef=model.coef_.tolist()
    LR_b=model.intercept_.tolist()
    return (LR_coef,LR_b)


# In[8]:


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


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(x_best_set_1,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=LogisticRegression()

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_LR = accuracy_score(y_val,prediction)
print(acc_score_LR)


# In[5]:


sensitivity_array=[0.7260089682766394, 0.7696626730025701, 1.0774842802117621, 1.1984050078355657]
group_sensitivity_array=[0.7683730158730153, 0.817460317460319, 1.1395767195767217, 1.2700198412698354]


# In[6]:


#calculate the predicted result with added noise

def LR_noise_result(df,w,b,sen):
    array=[]
    for i in range(0,100):
        loc=0
        scale=sen
        noise_1= np.random.laplace(loc, scale, df.shape[1])
        noise_2=np.random.laplace(loc, scale, 1)

        LR_coef_noise=noise_1+w
        LR_b_noise=noise_2+b

        predicted_result_noise=[]
        for i in range(0,df.shape[0]):
            predicted_result_noise.append(np.dot(df.iloc[i].tolist(),LR_coef_noise)+LR_b_noise)

        predicted_result_noise=pd.DataFrame(predicted_result_noise)
        predicted_result_noise=predicted_result_noise.mask( predicted_result_noise< 0, 0)
        predicted_result_noise=predicted_result_noise.mask( predicted_result_noise>0, 1)
        diff_result_noise=(predicted_result_noise[0].tolist()-y).tolist()
        acc=1-(df.shape[0]-diff_result_noise.count(0))/df.shape[0]
        
        array.append(acc)
    result=sum(array)/len(array)        
    return (result)


# In[9]:


for j in range(0,len(adjust_feature_set)):
    if j==0:
        fea_drop=fea_im_seq
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]] 
    else:
        fea_drop=fea_im_seq[:-j]    #drop last features
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset
     
        print(LR_noise_result(fea_drop_train,LR_coefficient(fea_drop_train)[0][0],LR_coefficient(fea_drop_train)[1],sensitivity_array[j]))
        


# In[11]:


def plot_LR_noise_result(df,w,b,sen):
    array=[]
    for i in range(1,11):
        epsilon=i/10 
        array.append(LR_noise_result(df,w,b,sen/epsilon))
    return (array)


# In[15]:


y_LR_no_noise=[0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857,0.8357142857142857]


# In[16]:


y_LR_proposed=plot_LR_noise_result(x,LR_coefficient(x)[0][0],LR_coefficient(x)[1],sensitivity_array[0])
y_LR_zhu=plot_LR_noise_result(x_best_set_1,LR_coefficient(x_best_set_1)[0][0],LR_coefficient(x_best_set_1)[1],sensitivity_array[3])
y_LR_group=plot_LR_noise_result(x_best_set_1,LR_coefficient(x_best_set_1)[0][0],LR_coefficient(x_best_set_1)[1],group_sensitivity_array[3])


# In[26]:


x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.figure(figsize=(8, 5))
plt.plot(x_label,y_LR_no_noise,linewidth=2.0,color='magenta',markerfacecolor='magenta',marker='o',label='Non private scheme')
plt.plot(x_label,y_LR_proposed,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='Proposed scheme')
plt.plot(x_label,y_LR_zhu,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='Zhu scheme')
plt.plot(x_label,y_LR_group,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='Group scheme')


plt.xlabel("Epslion",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
#plt.title('Epsilon VS Accuracy')
plt.legend(loc='best',prop={'size':12})
plt.grid(True)
plt.tick_params(labelsize=15)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS Accuracy LR dataset1 .png")

