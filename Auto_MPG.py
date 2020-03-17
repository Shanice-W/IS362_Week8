#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_fwf('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', header=None)
df.head(10)


# In[3]:


column_name = ['mpg',
          'cylinders',
          'displacement',
          'horsepower',
          'weight',
          'acceleration',
          'model_yr',
          'origin',
          'car_name']
df.columns = column_name
df.head(10)


# In[4]:


df.horsepower = df.horsepower.replace('?', np.nan)
df.horsepower = pd.to_numeric(df.horsepower)
df.dtypes


# In[5]:


origins = {1: 'USA', 
          2: 'Asia',
          3: 'Europe'}

df.origin = df.origin.map(origins)
df.head()


# In[6]:


a = sns.countplot(x="cylinders", hue='origin' ,data=df)
plt.xlabel("Cylinders",size = 20)
plt.ylabel("Count",size = 20)


# In[7]:


b = sns.regplot(x="horsepower", y="weight", data=df, color='m')
plt.title('The relationship between horsepower and weight', fontsize=20)
plt.xlabel("Horsepower",size = 15)
plt.ylabel("Weight",size = 15)


# In[8]:


c = sns.lmplot(x='model_yr', y='mpg', hue='origin', data=df)
plt.title('MPG over Time by Origin', fontsize=22)
plt.xlabel('Model Year', fontsize=15)
plt.ylabel('Miles Per Gallon', fontsize=15)

