
# coding: utf-8

# In[13]:


import networkx as nx 
import matplotlib.pyplot as plt
import community
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import numpy as np


# ### https://blog.csdn.net/Eastmount/article/details/78452581

# ### 100个节点
# ### https://networkx.github.io/documentation/networkx-1.10/examples/index.html 

# 1.n×n对称矩阵  √
# 
# 2.根据对称矩阵生成每个Agent pay-off  √
# 
# 3.Redistribution 根据pay-off生成New pay-off以及New合作矩阵
# 
# 4.迭代。计算最终cooperation rate

# R = 2 cooperation  
# P = 2 defection
# 
# C==>S  D==>T
# 
# R = 1, P = 0, S = 1 − T, 1 < T ≤ 2

# In[ ]:


def generateEdges(number,Z=4):
    Matrix = np.zeros([number,number]) 
    edges = []
    while np.sum(Matrix)!=Z*number:
        Matrix = np.zeros([number,number])  
        edges = []
        for i in range(number):
            degrees = np.sum(Matrix,0)
            degree = degrees[i]
            if degree<Z:
                degrees[i] = Z
                indexs = list(np.argwhere(degrees<Z))
                if(len(indexs)<int(Z-degree)):
                    continue
                index = random.sample(indexs,int(Z-degree))

                for ind in index:
                    Matrix[i][ind[0]] = 1
                    Matrix[ind[0]][i] = 1
                    edges += [(i,ind[0])]
                if np.sum(Matrix[i])>Z:
                    print("Error",i,degree)
            elif degree>Z:
                print("Error ",i,degree)
    return Matrix,edges


# In[3]:


def getPayoff(Matrix,AgentStrategy,T):
    R = 1.
    P = 0
    S = 1-T
    payoff = []
    for i in range(len(AgentStrategy)):
        arg = list(np.reshape(np.argwhere(Matrix[i]==1),[1,-1])[0])
        niC = list(AgentStrategy[arg]).count('r')
        niD = list(AgentStrategy[arg]).count('b')
        if AgentStrategy[i] == 'r':
            Omiga = 1
        else:
            Omiga = 0
        payoff += [niC*T+Omiga*(1-T)*(niC+niD)]
    return np.array(payoff)


# In[4]:


def updateStrategy(fit,Matrix,Strategy):
    fitMatrix = np.tile(fit,[len(fit),1])*Matrix
#     print(fitMatrix.shape)
    arg = np.argmax(fitMatrix,1)
    s = []
    for val in arg:
        s.append(Strategy[val])
    return np.array(s)


# In[5]:


def getfitness(alpha,theta,z,payoff,Matrix):
    f = payoff*0.0
    pay = payoff.copy()
    pay[np.argwhere(payoff<theta)]=theta
    
    for i,val in enumerate(payoff):  
        f[i] = np.max((1-alpha)*(pay[i]-theta),0)+ np.sum(Matrix[i]*alpha*(np.array(pay)-theta))/z
    return f


# In[6]:


import random 
import numpy as np
import sys

number = 1000       #总人数
proportion = 0.5 #合作者的比例  "r"=cooperator，‘b’=Defect
Matrix,Edges = generateEdges(number)


# In[7]:


result = []
for alph in np.linspace(1.0,0.0,num=51):
    rowCoperation = []
    for t in np.linspace(1.0,2.0,num=51):
        iteration = 10
        AgentStrategy = np.array(['r']*round(number*proportion)+['b']*round(number*(1-proportion)))
        for i in range(iteration):
            payoff = getPayoff(Matrix,AgentStrategy,T=t)
            f = getfitness(alpha=alph,theta=1.0,z=4,payoff=payoff,Matrix=Matrix)
            AgentStrategy = updateStrategy(f,Matrix,AgentStrategy)
        rowCoperation += [(list(AgentStrategy).count('r')/len(AgentStrategy))]
        sys.stdout.write('\r>> %d/51   %d/51 ' % (round(alph/0.02),round((t-1)/0.02)))  
        sys.stdout.flush()
    result.append(rowCoperation)
    


# 热力图

# In[8]:


from pylab import *
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

xmajorLocator   = MultipleLocator(0.1) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
xminorLocator   = MultipleLocator(0.02) #将x轴次刻度标签设置为5的倍数
 
ymajorLocator   = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
yminorLocator   = MultipleLocator(0.02) #将此y轴次刻度标签设置为0.1的倍数
 

fig, ax = plt.subplots(figsize = (6,4))

sns.heatmap(result, annot=False, vmax=1.,vmin = 0., xticklabels= True, yticklabels= False,linewidths=.01, square=False, cmap=plt.cm.Oranges,
           cbar_kws={'label': 'Level of cooperation'})


ax.set_xticks(np.linspace(1.0,51.0,num=11), minor=False)
ax.set_yticks(np.linspace(0.0,51.0,num=11), minor=False)

ax.set_xticklabels(np.round(np.linspace(1.0,2.0,num=11),2), minor=False)
ax.set_yticklabels(np.round(np.linspace(0.0,1.0,num=11),2), minor=False)
ax.invert_yaxis()
plt.xlabel('\nTemptation parameter,T',font1) 
plt.ylabel(r'Level of taxation , $\alpha$',font1) 

plt.title("Homogeneous\n"+r'$[\theta=1.0]$',font2) 

# plt.title(r'Microstrain [$mu epsilon$]')

fig.savefig("result.png",dpi=300,bbox_inches='tight',transparent=True)
show()


# In[9]:


result2 = result.copy()

result2.reverse()


# In[10]:


from pylab import *
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

xmajorLocator   = MultipleLocator(0.1) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
xminorLocator   = MultipleLocator(0.02) #将x轴次刻度标签设置为5的倍数
 
ymajorLocator   = MultipleLocator(0.1) #将y轴主刻度标签设置为0.5的倍数
ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
yminorLocator   = MultipleLocator(0.02) #将此y轴次刻度标签设置为0.1的倍数
 

fig, ax = plt.subplots(figsize = (6,4))

sns.heatmap(result2, annot=False, vmax=1.,vmin = 0., xticklabels= True, yticklabels= False,linewidths=.01, square=False, cmap=plt.cm.Oranges,
           cbar_kws={'label': 'Level of cooperation'})


ax.set_xticks(np.linspace(1.0,51.0,num=11), minor=False)
ax.set_yticks(np.linspace(0.0,51.0,num=11), minor=False)

ax.set_xticklabels(np.round(np.linspace(1.0,2.0,num=11),2), minor=False)
ax.set_yticklabels(np.round(np.linspace(0.0,1.0,num=11),2), minor=False)
ax.invert_yaxis()
plt.xlabel('\nTemptation parameter,T',font1) 
plt.ylabel(r'Level of taxation , $\alpha$',font1) 

plt.title("Homogeneous\n"+r'$[\theta=1.0]$',font2) 

# plt.title(r'Microstrain [$mu epsilon$]')

fig.savefig("result2.png",dpi=300,bbox_inches='tight',transparent=True)
show()


# In[11]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8.5,4.5))
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.xlim([0.98,2.02])
plt.ylim([-0.02,1.02])

plt.xlabel('Temptation parameter,T',font1) 
plt.ylabel('level of cooperation',font1) 

x = np.linspace(1.0,2.0,num=51)
plt.xticks(np.linspace(1.0,2.0,num=11)) 
plt.plot(x,result,'r--',marker='*')


# In[12]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8.5,4.5))
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

plt.xlim([0.98,2.02])
plt.ylim([-0.02,1.02])

plt.xlabel('Temptation parameter,T',font1) 
plt.ylabel('level of cooperation',font1) 

x = np.linspace(1.0,2.0,num=51)
plt.xticks(np.linspace(1.0,2.0,num=11)) 
plt.plot(x,result,'r--',marker='*')


# In[13]:


# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
# matplotlib.rcParams['font.family']='sans-serif'

# nodelist=[i for i in range(number)]
# plt.figure( figsize=(12,8),)
# G = nx.Graph()
# G.add_edges_from(Edges)
# # nx.draw(G,with_labels=True,pos=nx.random_layout(G),font_size=12,node_size=2000,node_color=colors) #alpha=0.3
# # pos=nx.spring_layout(G,iterations=50)
# pos=nx.random_layout(G)
# nx.draw_networkx_nodes(G, pos, alpha=1,nodelist=nodelist,edgecolors='k',node_size=300,node_color=AgentStrategy)
# nx.draw_networkx_edges(G, pos, alpha=0.5) #style='dashed'
# a = nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1) #font_size=5

# plt.axis('off')

