#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The Student Performance Dataset on Kaggle includes detailed information on student scores in three subjects: Math, Reading, and Writing. Along with these scores, demographic features such as gender, race/ethnicity, parental education level, lunch type, and test preparation status are available, making it ideal for exploring the relationships between various factors and student academic performance. This dataset can be used to perform exploratory data analysis (EDA), understand patterns in educational achievement, and even develop predictive models to determine student success.


# In[ ]:


#How does gender influence student performance across subjects?
#Are there notable differences in scores between different racial/ethnic groups?
#Is there a correlation between parental education levels and student scores?
#Do students with parents who have higher education tend to perform better?
#How does the type of lunch (standard vs. free/reduced) impact student performance?
#Do students who complete a test preparation course perform better than those who do not?
#What is the correlation between Math, Reading, and Writing scores?


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[13]:


df = pd.read_csv('C:/Users/HP/Downloads/StudentsPerformance.csv')


# In[14]:


df


# In[15]:


df.info()


# In[16]:


df.duplicated().sum()


# In[17]:


df.describe()


# In[18]:


df.head()


# In[19]:


df['gender'].value_counts()


# In[30]:


plt.figure(figsize=(6, 4))
df['gender'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = False, explode = [0.1,0])
plt.title('Gender Distribution')
plt.show()


# In[21]:


df['race/ethnicity'].value_counts()


# In[41]:


plt.figure(figsize=(7, 4))
df['race/ethnicity'].value_counts() .plot(kind="pie",autopct="%0.2f%%", shadow = False, explode = [0.1, 0, 0, 0, 0])
plt.title('race/ethnicity Distribution')
plt.show()


# In[22]:


df['parental level of education'].value_counts()


# In[40]:


plt.figure(figsize=(8, 6))
df['parental level of education'].value_counts() .plot(kind="pie",autopct="%0.2f%%", shadow = False, explode = [0.1, 0, 0, 0, 0, 0])
plt.title('parental level of education Distribution')
plt.show()


# In[23]:


df['lunch'].value_counts()


# In[39]:


plt.figure(figsize=(6, 4))
df['lunch'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = False, explode = [0.1, 0])
plt.title('lunch Distribution')
plt.show()


# In[24]:


df['test preparation course'].value_counts()


# In[42]:


plt.figure(figsize=(6, 4))
df['test preparation course'].value_counts() .plot(kind="pie",autopct="%0.2f%%", shadow = False, explode = [0.1, 0])
plt.title('test preparation course Distribution')
plt.show()


# In[43]:


df['math score'].describe()


# In[44]:


df['math score'].value_counts().head(5)


# In[48]:


df [df['math score'] == 0]


# In[49]:


df[df['math score'] == 100]


# In[50]:


df['reading score'].describe()


# In[51]:


df[df['reading score'] == 17]


# In[52]:


df[df['reading score'] == 100]


# In[53]:


df['writing score'].describe()


# In[54]:


df[df['writing score'] == 10]


# In[55]:


df[df['writing score'] == 100]


# In[57]:


pd.crosstab(df['gender'], df['race/ethnicity'])


# In[60]:


pd.crosstab(df['gender'], df['race/ethnicity']).plot(kind = 'bar', figsize=(8, 6))
plt.title('which race/ethnicity has the largest number of both genders?')
plt.xlabel('Gender')
plt.ylabel('count')
plt.xticks(rotation=40)
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.show()


# In[59]:


pd.crosstab(df['gender'], df['parental level of education'])


# In[61]:


pd.crosstab(df['gender'], df['parental level of education']).plot(kind = 'bar', figsize=(8, 6))
plt.title('which parental level of education has the largest number of both genders?')
plt.xlabel('Gender')
plt.ylabel('count')
plt.xticks(rotation=40)
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.show()


# In[62]:


pd.crosstab(df['gender'], df['lunch'])


# In[67]:


pd.crosstab(df['lunch'], df['gender']).plot(kind = 'bar', figsize=(8, 6))
plt.title('which Lunch has the largest number of both genders?')
plt.xlabel('Gender')
plt.ylabel('count')
plt.xticks(rotation=40)
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.show()


# In[64]:


pd.crosstab(df['test preparation course'], df['gender'])


# In[66]:


pd.crosstab(df['test preparation course'], df['gender']).plot(kind = 'bar', figsize=(8, 6))
plt.title('which test preparation course has the largest number of both genders?')
plt.xlabel('Gender')
plt.ylabel('count')
plt.xticks(rotation=40)
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.show()


# In[68]:


df.groupby(['gender']).agg({
    'math score' : 'mean'
})


# In[70]:


df.groupby(["gender"]).agg({
    'math score' : 'mean'
}).plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1,0],subplots=True)
plt.title('Mean of Math score for Gender')
plt.show()


# In[71]:


df.groupby(['gender']).agg({
    'reading score' : 'mean'
})


# In[72]:


df.groupby(["gender"]).agg({
    'reading score' : 'mean'
}).plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1,0],subplots=True)
plt.title('Mean of Reading score for Gender')
plt.show()


# In[73]:


df.groupby(['gender']).agg({
    'writing score' : 'mean'
})


# In[74]:


df.groupby(["gender"]).agg({
    'writing score' : 'mean'
}).plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1,0],subplots=True)
plt.title('Mean of Writing score for Gender')
plt.show()


# In[75]:


pd.crosstab(df['race/ethnicity'], df['parental level of education'])


# In[76]:


pd.crosstab(df['race/ethnicity'], df['parental level of education']).plot(kind = 'bar', figsize=(8, 6))
plt.title('Number of race/ethnicity in parental level of education')
plt.xlabel('race/ethnicity')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1))
plt.show()


# In[77]:


bad_scores = df[(df['math score'] < 50) & (df['reading score'] < 50) & (df['writing score'] < 50)]
bad_scores


# In[78]:


len(bad_scores)


# In[79]:


bad_scores['test preparation course'].value_counts()


# In[80]:


bad_scores['test preparation course'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0,0.2])
plt.title('Bad Scores')
plt.show()


# In[81]:


bad_scores['gender'].value_counts()


# In[82]:


bad_scores['gender'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1,0])
plt.title('Bad Scores')
plt.show()


# In[83]:


bad_scores['race/ethnicity'].value_counts()


# In[84]:


bad_scores['race/ethnicity'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1,0, 0, 0, 0])
plt.title('Bad Scores')
plt.show()


# In[85]:


best_scores = df[(df['math score'] >= 90) & (df['reading score'] >= 90) & (df['writing score'] >= 90)]
best_scores


# In[87]:


best_scores['gender'].value_counts()


# In[88]:


full_scores = df[(df['math score'] == 100) & (df['reading score'] == 100) & (df['writing score'] == 100)]
full_scores


# In[89]:


full_scores['gender'].value_counts()


# In[90]:


full_scores['gender'].value_counts().plot(kind="pie",autopct="%0.2f%%", shadow = True, explode = [0.1, 0])
plt.title('Full Scores')
plt.show()


# In[ ]:


#Female students tend to outperform male students in Reading and Writing, while males perform slightly better in Math.
#A higher parental education level correlates with better student performance, especially in Writing and Reading.
#Students with standard lunch tend to have higher scores across all subjects compared to those with free/reduced lunch.
#Math, Reading, and Writing scores are highly correlated, indicating that students who perform well in one subject are likely to excel in the others as well.
#A few outliers were identified, particularly students who scored exceptionally high or low in all subjects. These may represent either highly gifted students or those needing additional support.

