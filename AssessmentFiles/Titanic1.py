# coding: utf-8

# # Austin Duling
# ## November 21, 2020

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("titanic.csv")
df


# # Question 1

# ## What is the likelyhood that a third class child < 10 survived the crash?

# In[4]:


df1 = df.loc[((df['Age'] <= 10) & (df['Pclass'] == 3))]
df1


# In[76]:


survived_probs = df1.groupby('Survived').size().div(len(df1))
survived_probs


# In[13]:


labels = ['Did not survive', 'Survived']
fig, ax = plt.subplots()
ax.pie(survived_probs, autopct='%1.1f%%',labels = labels, shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# In "survived_probs" we see the probability of a 3rd class child under the age of 10 surviving the Titanic crash. It was almost a 60/40 chance probability against survival for this demographic.

# # Question 2

# ## What was the average cost of admission for the Titanic?

# In[32]:


avg_cost = df['Fare'].mean()


# In[70]:


print('The average cost to board the Titanic was $' + str(avg_cost) + '.' )


# # Question 3

# ## What was the average cost of admission by social class?

# In[34]:


df1 = df[['Fare', 'Pclass']]


# In[69]:


data = []
df2 = df1.loc[((df['Pclass'] == 1))] #Finding avg for first class ticket
first_class_ticket = df2['Fare'].mean()
data.append(first_class_ticket)
df3 = df1.loc[((df['Pclass'] == 2))] #Finding avg for second class ticket
second_class_ticket = df3['Fare'].mean()
data.append(second_class_ticket)
df4 = df1.loc[((df['Pclass'] == 3))] #Finding avg for third class ticket
third_class_ticket = df4['Fare'].mean()
data.append(third_class_ticket)

print('The average cost for a first class ticket was: $' + str(first_class_ticket))
print('The average cost for a second class ticket was: $' + str(second_class_ticket))
print('The average cost for a third class ticket was: $' + str(third_class_ticket))


categories = ['First Class', 'Second Class', 'Third Class']
height = 0.6
fig, ax = plt.subplots()

y_pos = np.arange(len(categories))

ax.barh(y_pos, data, align='center',color = 'g', height = height)
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.invert_yaxis()
ax.set_xlabel('Average cost of ticket per class per $ ')


plt.show()


# In[ ]:




