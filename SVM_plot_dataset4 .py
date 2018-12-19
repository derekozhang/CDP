
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import matplotlib
import time 
from subprocess import check_output
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from matplotlib import gridspec
import numpy as np
import numpy
import sys
import warnings
from sklearn.model_selection import train_test_split
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.simplefilter("ignore")
data = pd.read_csv('x_train_37.csv')
y= pd.read_csv('y_train.csv')
y=y.drop('0',axis=1)
data=data.drop('Unnamed: 0',axis=1)
df_scaled = preprocessing.scale(data) #z-score
x=pd.DataFrame(df_scaled)  #array transfer into dataframe
line = pd.DataFrame({"0.1": 0}, index=[0])
y = pd.concat([line,y],ignore_index=True)
y=y['0.1']


# In[2]:


sensitivity_array=[0.42912264379180115, 0.3291189879010309, 0.24793080459497377, 0.2858057685068165, 0.3357716236639724, 0.3584983912007751, 0.30329103229305476, 0.39134662503115974, 0.3732019114743513, 0.3888006988496443, 0.3371725877597541, 0.36846113563234856, 0.56715990786762, 0.6288385864517355, 0.7940738894847562, 0.7222110311669152, 0.7334812953205717, 0.9414997815520915, 0.5731954964855278, 0.7068841249153495, 0.6567239313611217, 0.8244337562304037, 1.1070901342421402]

group_sensitivity_array=[0.577150537634411, 0.4422043010752552, 0.33467741935483075, 0.38168682795697484, 0.4515591397849176, 0.46839157706093304, 0.4027457757296543, 0.5215557795699152, 0.4927419354838978, 0.5241666666667075, 0.4462243401760073, 0.48681675627249, 0.7429487179488328, 0.8364055299540385, 1.04099462365608, 0.9474462365592666, 0.9497153700191241, 1.2482078853048428, 0.746434634974674, 0.9395698924733056, 0.8625000000001858, 1.0786290322582377, 1.4465287517533907]
best_feature_set_1=[0, 6, 15, 3, 17, 7, 2, 13, 21, 19, 33, 35, 5, 4]
adjust_feature_set=[1, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
x_best_set_1=x[best_feature_set_1]


# In[4]:


from sklearn.svm import SVC
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
split_size=0.3

X_train, X_test, y_train, y_test = train_test_split( x_best_set_1,y,test_size=split_size,random_state=22)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

model=SVC(kernel = 'linear')

kfold = model_selection.KFold(n_splits=5,random_state=7)
score=model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc_score_svm = accuracy_score(y_val,prediction)
print(acc_score_svm)


# In[7]:


from sklearn.svm import SVC
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
split_size=0.3

#calculate the w and b in the logistic regression
def svm_coefficient(df):

    X_train, X_test, y_train, y_test = train_test_split( df,y,test_size=split_size,random_state=22)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

    model=SVC(kernel = 'linear')

    kfold = model_selection.KFold(n_splits=5,random_state=7)
    score=model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    acc_score_LR = accuracy_score(y_val,prediction)
    #print(acc_score_LR)
    svm_coef=model.coef_.tolist()
    svm_b=model.intercept_.tolist()
    return (svm_coef,svm_b)


# In[8]:


def svm_noise_result(df,w,b,sen):
    array=[]
    for i in range(0,500):
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
        
        array.append(acc)
    result=sum(array)/len(array)        
    return (result)


# In[37]:


svm_noise_result(x,svm_coefficient(x)[0][0],svm_coefficient(x)[1],sensitivity_array[0])


# In[9]:


x_best_feature_set_2=[0, 6, 15, 3, 17, 7, 2, 13, 21, 19, 33, 35, 5, 4,1, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28]
x_best_set_2=x[x_best_feature_set_2]


# In[10]:


def plot_svm_noise_result(df,w,b,sen):
    array=[]
    for i in range(1,11):
        epsilon=i/10 
        array.append(svm_noise_result(df,w,b,sen/epsilon))
    return (array)


# In[39]:


y_svm_proposed=plot_svm_noise_result(x_best_set_2,svm_coefficient(x_best_set_2)[0][0],svm_coefficient(x_best_set_2)[1],sensitivity_array[0])
y_svm_zhu=plot_svm_noise_result(x_best_set_1,svm_coefficient(x_best_set_1)[0][0],svm_coefficient(x_best_set_1)[1],sensitivity_array[16])
y_svm_group=plot_svm_noise_result(x_best_set_1,svm_coefficient(x_best_set_1)[0][0],svm_coefficient(x_best_set_1)[1],group_sensitivity_array[16])


# In[40]:


y_svm_no_noise=[0.883,0.883,0.883,0.883,0.883,0.883,0.883,0.883,0.883,0.883,]


# In[46]:


x_label=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
plt.figure(figsize=(8, 6))
#plt.plot(x_label,y_svm_proposed,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='proposed scheme')
#plt.plot(x_label,y_svm_group,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='group scheme')
#plt.plot(x_label,y_svm_zhu,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='zhu scheme')
#plt.plot(x_label,y_svm_no_noise,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='No noise scheme')
plt.plot(x_label,y_svm_no_noise,linewidth=2.0,color='magenta',markerfacecolor='magenta',marker='o',label='Non private scheme')
plt.plot(x_label,y_svm_proposed,linewidth=2.0,color='red',markerfacecolor='red',marker='o',label='Proposed scheme')

plt.plot(x_label,y_svm_zhu,linewidth=2.0,color='blue',markerfacecolor='blue',marker='o',label='Zhu scheme')
plt.plot(x_label,y_svm_group,linewidth=2.0,color='green',markerfacecolor='green',marker='o',label='Group scheme')
plt.xlabel("Epslion",fontsize=20)
plt.ylabel("Accuracy",fontsize=20)
#plt.title('Epsilon VS Accuracy')
plt.legend(loc='best',prop={'size':16})
plt.grid(True)
plt.tick_params(labelsize=20)
#xlim(0.2, 1)
#ylim(0,1)
#plt.show()
plt.savefig("Epsilon VS Accuracy SVM dataset2 .png")

