#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


diabetes = datasets.load_diabetes()


# In[3]:


diabetes


# In[4]:


print(diabetes.DESCR)


# In[5]:


print(diabetes.feature_names)


# In[6]:


X = diabetes.data
Y = diabetes.target


# In[7]:


X.shape, Y.shape


# In[8]:


X, Y = datasets.load_diabetes(return_X_y=True)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[11]:


X_train.shape, Y_train.shape


# In[12]:


X_test.shape, Y_test.shape


# In[13]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[14]:


model = linear_model.LinearRegression()


# In[15]:


model.fit(X_train, Y_train)


# In[16]:


Y_pred = model.predict(X_test)


# In[17]:



print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


# In[20]:


print(diabetes.feature_names)


# In[21]:


r2_score(Y_test, Y_pred)


# In[22]:


r2_score(Y_test, Y_pred).dtype


# In[23]:


'%f' % 0.523810833536016


# In[24]:


'%.3f' % 0.523810833536016


# In[25]:


'%.2f' % 0.523810833536016


# In[26]:


import seaborn as sns


# In[27]:


Y_test


# In[28]:


import numpy as np
np.array(Y_test)


# In[29]:


Y_pred


# In[30]:


sns.scatterplot(Y_test, Y_pred)


# In[31]:


sns.scatterplot(Y_test, Y_pred, marker="+")


# In[32]:


sns.scatterplot(Y_test, Y_pred, alpha=0.5)


# In[ ]:




