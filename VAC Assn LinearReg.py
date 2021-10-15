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


X.shape, Y.shape


# In[10]:


import pandas as pd


# In[11]:


BostonHousing = pd.read_csv("BostonHousing.csv")
BostonHousing


# In[13]:


Y = BostonHousing.medv
Y


# In[15]:


X = BostonHousing.drop(['medv'], axis=1)
X


# In[16]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[19]:


X_train.shape, Y_train.shape


# In[20]:


X_test.shape, Y_test.shape


# In[27]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[28]:


model = linear_model.LinearRegression()


# In[29]:


model.fit(X_train, Y_train)


# In[30]:


Y_pred = model.predict(X_test)


# In[31]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


# In[32]:


r2_score(Y_test, Y_pred)


# In[34]:


r2_score(Y_test, Y_pred).dtype


# In[36]:


'%f' % 0.523810833536016


# In[38]:


'%.3f' % 0.523810833536016


# In[40]:


'%.2f' % 0.523810833536016


# In[41]:


import seaborn as sns


# In[43]:


Y_test


# In[45]:


import numpy as np
np.array(Y_test)


# In[47]:


Y_pred


# In[49]:


sns.scatterplot(Y_test, Y_pred)


# In[51]:


sns.scatterplot(Y_test, Y_pred, marker="+")


# In[52]:


sns.scatterplot(Y_test, Y_pred, alpha=0.5)

