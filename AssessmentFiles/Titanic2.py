#!/usr/bin/env python
# coding: utf-8

# # Austin Duling
# ## November 25, 2020

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("titanic.csv")


# # Section 1

# In[3]:


df1 = df.loc[((df['Sex'] == 'male') & (df['Survived'] <= 1))] # Dataframe for men and survived columns

male_survived_probs = df1.groupby('Survived').size().div(len(df1))


# In[4]:


df2 = df.loc[((df['Sex'] == 'female') & (df['Survived'] <= 1))] # Dataframe for women and survived columns
df2
female_survived_probs = df2.groupby('Survived').size().div(len(df2))


# In[5]:



fig = plt.figure(figsize=(6,4.8)) 
fig.set_tight_layout(True)

ax = fig.add_subplot(211, xlabel = 'Men') # Subplot for men's results
labels = ['Did not survive', 'Survived']
ax.pie(male_survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1 = fig.add_subplot(212, xlabel = 'Women') # Subplot for women's results
labels = ['Did not survive', 'Survived']
ax1.pie(female_survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# As could be assumed from the widely known telling of the events of the Titanic, the passengers followed the "Women and children off the boat first" rule. The evidence of this is shown in the data here, where most women survived and most men did not. 

# # Section 2

# In[6]:


df1 = df.loc[((df['Pclass'] == 1) & (df['Survived'] <= 1))] # Dataframe for First and survived columns
first_class_survived_probs = df1.groupby('Survived').size().div(len(df1))


# In[7]:


df2 = df.loc[((df['Pclass'] == 2) & (df['Survived'] <= 1))] # Dataframe for Second class and survived columns
second_class_survived_probs = df2.groupby('Survived').size().div(len(df2))


# In[8]:


df3 = df.loc[((df['Pclass'] == 3) & (df['Survived'] <= 1))] # Dataframe for Third class and survived columns
third_class_survived_probs = df3.groupby('Survived').size().div(len(df3))


# In[9]:


fig = plt.figure(figsize=(7,7)) 
fig.set_tight_layout(True)

ax = fig.add_subplot(311, xlabel = 'First Class Survived') # Subplot for first class results
labels = ['Did not survive', 'Survived']
ax.pie(first_class_survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1 = fig.add_subplot(312, xlabel = 'Second Class Survived') # Subplot for second class results
labels = ['Did not survive', 'Survived']
ax1.pie(second_class_survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax2 = fig.add_subplot(313, xlabel = 'Third Class Survived') # Subplot for third class results
labels = ['Did not survive', 'Survived']
ax2.pie(third_class_survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# For this section, a fairly obvious trend is found in the impact class had on your chance of survival on the Titanic. Put simply, the wealthier you were, the better chance you had of living.

# # Question 3 - Female by class

# In[37]:


# This section will be finding the conditional probability that a first class female will survive. 
df1 = df.loc[((df['Sex'] == 'female') & (df['Survived'] <= 1) & (df['Pclass'] == 1))] 


# In[16]:


first_class_female_survived_probs = df1.groupby('Survived').size().div(len(df1))
first_class_female_survived_probs


# Here it is shown that almost all of the women that were riding first class surived. Only 3 of the 94 women in first class died, so simplifying the ratio 3:94 would give us a  1:~31 chance of dying for this demographic.

# In[36]:


# This section will be finding the conditional probability that a second class female will survive. 
df2 = df.loc[((df['Sex'] == 'female') & (df['Survived'] <= 1) & (df['Pclass'] == 2))] 


# In[21]:


second_class_female_survived_probs = df2.groupby('Survived').size().div(len(df2))
second_class_female_survived_probs


# From this data we can gather that 6 of the 76 women riding second class died, so a second class women would have just a little over 92% probable chance of dying aboard the Titanic. 

# In[35]:


# This section will be finding the conditional probability that a third class female will survive. 
df3 = df.loc[((df['Sex'] == 'female') & (df['Survived'] == 0) & (df['Pclass'] == 3))] 


# In[23]:


third_class_female_survived_probs = df3.groupby('Survived').size().div(len(df3))
third_class_female_survived_probs


# Interestingly enough, the third class female demographic had exactly the same number of fatalities as the number of passengers survived. This leaves a third class female with a 50% chance of survival, or an even split. 

# # Question 3 - Male by class

# In[25]:


# This section will be finding the conditional probability that a first class male will survive. 
df1 = df.loc[((df['Sex'] == 'male') & (df['Survived'] <= 1) & (df['Pclass'] == 1))] 
df1


# In[26]:


first_class_male_survived_probs = df1.groupby('Survived').size().div(len(df1))
first_class_male_survived_probs


# Given the data gained, we can collect that 77 of the males in first class died while only 45 survived. This puts the first class male at having around a 36% chance of survival. Much lower than any of the women demographics so far already. 

# In[28]:


# This section will be finding the conditional probability that a second class male will survive. 
df2 = df.loc[((df['Sex'] == 'male') & (df['Survived'] <= 1) & (df['Pclass'] == 2))] 
df2


# In[31]:


second_class_male_survived_probs = df2.groupby('Survived').size().div(len(df2))
second_class_male_survived_probs


# The second class male had a lower chance of survival than the first class male, with 91 out of the 108 in this demographic dying. This lowers the odds of survival for the second class male to roughly 16% chance of survival.

# In[33]:


# This section will be finding the conditional probability that a third class male will survive. 
df3 = df.loc[((df['Sex'] == 'male') & (df['Survived'] <= 1) & (df['Pclass'] == 3))] 
df3


# In[34]:


third_class_male_survived_probs = df3.groupby('Survived').size().div(len(df3))
third_class_male_survived_probs


# The odds of survival seen here are not much different than seen in second class men. 295 out of 343 third class men died on the Titanic, giving them a roughly 14% chance of survival.

# In[ ]:




