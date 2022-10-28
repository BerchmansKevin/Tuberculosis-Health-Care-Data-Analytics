#!/usr/bin/env python
# coding: utf-8

# ## HEALTHCARE DATA ANALYTICS

# ## `Topic : TB (Tuberculosis)`

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


tb = pd.read_csv("Tb disease symptoms.csv")


# In[3]:


tb


# In[4]:


tb.head()


# In[5]:


tb.tail()


# In[6]:


tb.shape


# In[7]:


tb.size


# In[8]:


tb.dtypes


# In[9]:


type(tb)


# In[10]:


tb.isnull().head()


# In[11]:


tb.describe()


# In[12]:


tb['chest_pain'].value_counts()


# In[13]:


tb['coughing_blood'].value_counts()


# In[14]:


tb.groupby('body feels tired').mean()


# In[15]:


tb.groupby('chest_pain').mean()


# In[16]:


tb.groupby('coughing_blood').mean()


# In[17]:


X = tb.drop("loss_of_appetite",axis=1)
X


# In[18]:


y = tb["loss_of_appetite"]
y


# # Split Data (Train and Test)

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[21]:


X_train


# In[22]:


X_train.shape


# In[23]:


type(X_train)


# In[24]:


X_test 


# In[25]:


X_test.shape


# In[26]:


y_train


# In[27]:


y_train.shape


# In[28]:


y_test


# In[29]:


y_test.shape


# # Create LogisticRegression Model (Fit and Predict)

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


lor = LogisticRegression()


# In[32]:


lor.fit(X_train, y_train)


# In[33]:


y_predict = lor.predict(X_test)
y_predict


# # Feature Scaling using MinMaxScaler

# In[34]:


from sklearn.preprocessing import MinMaxScaler


# In[35]:


scaler = MinMaxScaler()


# In[36]:


ss3 = scaler.fit_transform(X_train)
ss3


# In[37]:


m_ss1 = scaler.transform(X_test)
m_ss1


# # Precision Score

# In[38]:


from sklearn.metrics import precision_score 
print(precision_score(y_test, y_predict))


# # Recall Score

# In[39]:


from sklearn . metrics import recall_score 
print(recall_score(y_test, y_predict))


# # Accuracy Score

# In[40]:


from sklearn.metrics import accuracy_score 
lor_ascore = accuracy_score(y_test, y_predict)


# In[41]:


lor_ascore


# # Confusion Matrix

# In[42]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_predict) 
cnf_matrix


# In[43]:


cf_ac_score = accuracy_score(y_test, y_predict)
cf_ac_score


# # TB Plot 

# In[44]:


tb.plot()


# # Create Scatter Plot

# In[45]:


plt.scatter(y_test, y_predict, color='red')
plt.show()


# In[46]:


pd.crosstab(tb.coughing_blood,tb.chest_pain).plot(kind="bar",figsize=(19,8),color=['blue','hotpink'])
plt.show()


# In[47]:


sns.heatmap(tb.head(), annot=True)


# # Heatmap of Confusion Matrix 

# In[48]:


sns.heatmap(pd.DataFrame(tb), cmap='gist_rainbow_r', annot=True)


# # Heatmap of Confusion Matrix 

# In[49]:


sns.heatmap(confusion_matrix(y_test,y_predict) / len(y), cmap='gist_rainbow' , annot=True)


# # PAIRPLOT

# In[50]:


sns.pairplot(X)


# # DISTPLOT

# In[51]:


sns.distplot(tb["swollen_lymph_nodes"], bins=5)


# In[ ]:




