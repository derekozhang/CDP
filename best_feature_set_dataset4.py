
# coding: utf-8

# In[50]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import matplotlib
import time 
from subprocess import check_output
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import numpy
import sys
import warnings
from sklearn.model_selection import train_test_split
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.simplefilter("ignore")
data = pd.read_csv('x_train.csv')
y= pd.read_csv('y_train.csv')
y=y.drop('0',axis=1)
data=data.drop('Unnamed: 0',axis=1)
df_scaled = preprocessing.scale(data) #z-score
x=pd.DataFrame(df_scaled)  #array transfer into dataframe
line = pd.DataFrame({"0.1": 0}, index=[0])
y = pd.concat([line,y],ignore_index=True)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
acc_score_array=[]
split_size=0.3
for i in range(0,20):

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

acc_score_mean=sum(acc_score_array)/len(acc_score_array)
print(acc_score_mean)


# In[46]:


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


# In[47]:


fea_sum_array=[]
for i in range(0,x.shape[1]):
    
    fea_drop=fea_im_seq[:-i]    #drop last features
    fea_sum_array.append(fea_drop.sum())
    
fea_sum_array=pd.DataFrame(fea_sum_array)
fea_sum_array=fea_sum_array.replace(0,1)
fea_sum_array=fea_sum_array.iloc[:,1]

print(fea_sum_array)


# In[48]:


fea_ac_array=[]
fea_ac_matrix=[]
split_size=0.3     

for j in range(0,300):
    fea_ac_array=[]
    for i in range(0,x.shape[1]):
        if i==0:
            fea_drop=fea_im_seq    #drop last features
            fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset   
        else:
            fea_drop=fea_im_seq[:-i]    #drop last features
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


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(fea_sum_array,fea_ac_array,linewidth=3.0,color='red',markerfacecolor='blue',marker='o')
plt.xlabel("Feature importance",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
#plt.title('Accuracy VS Feature importance')
plt.grid(True)
plt.tick_params(labelsize=20)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Accuracy VS Feature importance Dataset4.png")


# In[ ]:


# find best feature set I and adjusted feature set
fea_ac_list=list(fea_ac_array)
fea_ac_position=fea_ac_list.index(max(fea_ac_list))
best_feature_set_1=fea_im_seq['feature'].tolist()[:(x.shape[1]-fea_ac_position)]
adjust_feature_set=list(set(x)-set(best_feature_set_1))
print(best_feature_set_1)
print(adjust_feature_set)


# In[ ]:


corr_p=[]
for i in range(0,x.shape[1]-2):
    
    if i==0:
        fea_drop=fea_im_seq    #drop last features
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset   
    else:

        fea_drop=fea_im_seq[:-i]    #drop last features
        fea_drop_train=x.loc[:,fea_drop.T.iloc[0]]  #training dataset
   
    df_pc= fea_drop_train.T.corr()
    df_pc=df_pc.replace(1,0)
    df_pc=abs(df_pc)
    df_pc=df_pc[df_pc>0.8]

    df_pc_sum=df_pc.sum()
    df_corr=df_pc_sum.max()
    corr_p.append(df_corr)
corr_p


# In[ ]:


plt.figure(figsize=(8, 6))
x_fea_im=np.arange(172)
x_fea_im=np.delete(x_fea_im,[1,0])
#x_fea_im=x_fea_im.tolist()
x_fea_im=np.flip(x_fea_im,axis=0)
x_fea_im=x_fea_im.tolist()
plt.plot(x_fea_im,corr_p,linewidth=3.0,color='red',markerfacecolor='blue',marker='o')

plt.xlabel('Number of features',fontsize=20)
plt.ylabel("Data correlation",fontsize=20)
#plt.title('Number of features VS Correlation')
#plt.legend(loc='best')
plt.grid()
plt.tick_params(labelsize=20)
#xlim(3, 13)
#ylim(0,1)
#plt.show()
plt.savefig("Number of features VS Correlation Dataset4.png")

