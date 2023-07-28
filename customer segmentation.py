#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sys
import warnings


# In[78]:


dataset=pd.read_excel("C:/Users/HOME/Desktop/Mall_Customers_.xlsx")


# In[79]:


dataset


# In[80]:


dataset.head()


# In[81]:


dataset.drop('CustomerID', axis=1, inplace = True)
dataset.head()


# In[82]:


dataset.shape


# In[83]:


dataset.info()


# In[84]:


dataset.isnull().sum()


# In[85]:


cor = dataset.corr()
sns.set(font_scale=1.4)
plt.figure(figsize=(9,8))
sns.heatmap(cor, annot=True, cmap='plasma')
plt.tight_layout()
plt.show()


# In[86]:


plt.figure(figsize=(16,12),facecolor='#9DF08E')

# Spending Score
plt.subplot(3,3,1)
plt.title('Spending Score\n', color='#FF000B')
sns.distplot(dataset['Spending Score (1-100)'], color='orange')

# Age
plt.subplot(3,3,2)
plt.title('Age\n', color='#FF000B')
sns.distplot(dataset['Age'], color='#577AFF')

# Annual Income 
plt.subplot(3,3,3)
plt.title('Annual Income\n', color='#FF000B')
sns.distplot(dataset['Annual Income (k$)'], color='black')

plt.suptitle(' Distribution Plots\n', color='#0000C1', size = 30)
plt.tight_layout()


# In[87]:


from sklearn.preprocessing import LabelEncoder

print('\033[0;32m' + 'Before Label Encoder\n' + '\033[0m' + '\033[0;32m', dataset['Gender'])

le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset.iloc[:,0])

print('\033[0;31m' + '\n\nAfter Label Encoder\n' + '\033[0m' + '\033[0;31m', dataset['Gender'])


# In[88]:


dataset.head()


# In[89]:


# Let's calculate how much to shop for which gender

spending_score_male = 0
spending_score_female = 0

for i in range(len(dataset)):
    if dataset['Gender'][i] == 1:
        spending_score_male = spending_score_male + dataset['Spending Score (1-100)'][i]
    if dataset['Gender'][i] == 0:
        spending_score_female = spending_score_female + dataset['Spending Score (1-100)'][i]


print('\033[1m' + '\033[93m' + f'Males Spending Score  : {spending_score_male}')
print('\033[1m' + '\033[93m' + f'Females Spending Score: {spending_score_female}')


# In[ ]:





# In[90]:


#clustering


# In[91]:


# Let's look at the relationship between Age and Spending score

plt.figure(figsize=(12,8))
sns.scatterplot(x = dataset['Age'], y = dataset['Spending Score (1-100)'])
plt.title('Age - Spending Score', size = 23, color='red')


# In[92]:


# x assignment
x = dataset.iloc[:,0:].values 
print("\033[1;31m"  + f'X data before PCA:\n {x[0:5]}')


# standardization before PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) 
X_2D = pca.fit_transform(X)
print("\033[0;32m" + f'\nX data after PCA:\n {X_2D[0:5,:]}')


# In[ ]:





# In[ ]:





# In[ ]:




