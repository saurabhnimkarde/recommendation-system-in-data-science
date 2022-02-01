#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


books = pd.read_csv("BOOKS1.csv")


# In[4]:


books


# In[5]:


books[0:5]


# In[6]:


books_df = books.rename({'User.ID':'userid','Book.Title':'booktitle','Book.Rating':'bookrating'},axis = 1)


# In[7]:


books_df


# In[8]:


books1 = books_df.drop(['Unnamed: 0'], axis = 1)


# In[9]:


books1


# In[10]:


len(books1['userid'].unique())


# In[11]:


array_user = books1['userid'].unique()


# In[12]:


len(books1['booktitle'].unique())


# In[13]:


books_df1 = books1.pivot_table(index = 'userid',
                        columns = 'booktitle',
                        values = 'bookrating').reset_index(drop = True)


# In[14]:


books_df1


# In[15]:


books_df1.index = books1.userid.unique()


# In[16]:


books_df1.fillna(0, inplace = True)


# In[17]:


books_df1


# In[18]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore")
user_sim = 1 - pairwise_distances(books_df1.values, metric = 'cosine')


# In[19]:


user_sim


# In[20]:


user_sim_df = pd.DataFrame(user_sim)
user_sim_df.iloc[0:5,0:5]       


# In[21]:


user_sim_df.index = books1.userid.unique()
user_sim_df.columns = books1.userid.unique()


# In[22]:


user_sim_df.iloc[0:5,0:5]


# In[23]:


np.fill_diagonal(user_sim,0)


# In[24]:


user_sim_df.idxmax(axis = 1)


# In[25]:


books1[(books1['userid'] == 162107) | (books1['userid'] == 276726)]


# In[26]:


books1[(books1['userid']==276729) | (books1['userid']==276726)]


# In[27]:


user_1 = books1[books1['userid'] == 276729]


# In[28]:


user_2 = books1[books1['userid'] == 276726]


# In[29]:


pd.merge(user_1,user_2, on = 'booktitle', how = 'outer')


# In[ ]:




