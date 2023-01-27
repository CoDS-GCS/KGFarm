#!/usr/bin/env python
# coding: utf-8

# ## KGFarm's feature transformation

# In[1]:


import pandas as pd
entity_df = pd.read_csv(r'C:\Users\niki_\Google Drive\GRAD SCHOOL\Papers\KGFarm-LFE\diabetes.csv')


# In[2]:


import os
import pandas as pd
os.chdir('../../')

from operations.api import KGFarm
kgfarm = KGFarm()


# In[3]:


kgfarm.recommend_cleaning_operations(entity_df)


# In[5]:


entity_df


# KGFarm exploits the abstracted link between the KGLiDS graph and Farm graph
#    * <code> kgfarm.recommend_feature_transformations</code> returns all possible set of feature transformations that exists in the database.
#    * <b> You can pass your existing <code>entity_df</code> to <code>kgfarm.recommend_feature_transformations(entity:pd.Dataframe)</code> to look for possible feature transformations for that very entity dataframe.

# In[4]:


transformation_info = kgfarm.recommend_feature_transformations(entity_df)
transformation_info


# In[5]:


entity_df, transformation_model = kgfarm.apply_transformation(transformation_info.iloc[0],entity_df)


# In[ ]:


entity_df


# In[ ]:


# entity_df['def'] = entity_df['def'].apply(lambda x: 1 if x == 'Y' else 0)
# entity_df.reset_index(drop=True, inplace=True)
# entity_df


# In[9]:


entity_df, transformation_model = kgfarm.apply_transformation(transformation_info.iloc[1],entity_df)


# In[ ]:


X, y = kgfarm.select_features(entity_df, dependent_variable="Outcome",select_by='correlation')
X


# In[10]:


dependent_variable = "Outcome"
independent_variables = [feature for feature in list(entity_df.columns) if feature != dependent_variable]

X = entity_df[independent_variables]
y = entity_df[dependent_variable]


# In[11]:


from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier(n_estimators=10000)
lrc = LogisticRegression()


# In[ ]:


k_folds = StratifiedKFold(n_splits = 10, shuffle=True)

scores = cross_val_score(rfc, X, y, cv = k_folds, scoring='f1').mean()
print(scores)
scores1 = cross_val_score(rfc, X, y, cv = k_folds, scoring='precision').mean()
print(scores1)
scores2 = cross_val_score(rfc, X, y, cv = k_folds, scoring='recall').mean()
print(scores2)
scores = cross_val_score(rfc, X, y, cv = k_folds, scoring='f1').mean()
scores


# In[ ]:


entity_df

