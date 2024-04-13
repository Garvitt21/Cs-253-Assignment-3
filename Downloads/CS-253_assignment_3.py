#!/usr/bin/env python
# coding: utf-8

# In[198]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[199]:


train_file = pd.read_csv('C:\\Users\\ASUS\\Downloads\\who-is-the-real-winner (1)\\train.csv')
test_file = pd.read_csv('C:\\Users\\ASUS\\Downloads\\who-is-the-real-winner (1)\\test.csv')


# In[200]:


le_constituency = LabelEncoder()
le_party = LabelEncoder()
le_state = LabelEncoder()
le_education = LabelEncoder()

train_file['Constituency ∇'] = le_constituency.fit_transform(train_file['Constituency ∇'])
train_file['Party'] = le_party.fit_transform(train_file['Party'])
train_file['state'] = le_state.fit_transform(train_file['state'])
train_file['Education'] = le_education.fit_transform(train_file['Education']) 
train_file['Total Assets'] = train_file['Total Assets'].str.split(' ').str[0].astype(int)
train_file['Liabilities'] = train_file['Liabilities'].str.split(' ').str[0].astype(int)


# In[201]:


X = train_file[['Constituency ∇','Party','Criminal Case', 'Total Assets', 'Liabilities','state']]
y = train_file['Education']


# In[202]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)


# In[203]:


le_constituency1 = LabelEncoder()
le_party1 = LabelEncoder()
le_state1 = LabelEncoder()
parties = test_file['Party'].unique()
test_file['Constituency ∇'] = le_constituency1.fit_transform(test_file['Constituency ∇'])
test_file['Party'] = le_party1.fit_transform(test_file['Party'])
test_file['state'] = le_state1.fit_transform(test_file['state'])
test_file['Total Assets'] = test_file['Total Assets'].str.split(' ').str[0].astype(int)
test_file['Liabilities'] = test_file['Liabilities'].str.split(' ').str[0].astype(int)


# In[204]:


X1 = test_file[['Constituency ∇','Party','Criminal Case', 'Total Assets', 'Liabilities','state']]
test_predictions = rf_model.predict(X1)
predicted_education_labels= le_education.inverse_transform(test_predictions)
submission_df = pd.DataFrame({'ID': test_file['ID'], 'Education': predicted_education_labels})
# Write the DataFrame to a CSV file, excluding other columns
print(submission_df)
submission_df.to_csv('C:\\Users\\ASUS\\Downloads\\sample.csv', index=False, columns=['ID', 'Education'])


# In[205]:


graph = pd.DataFrame({"Parties" : parties})
graph["Criminal Case"] = 0
graph["Wealth"] = 0 
graph['Count'] = 0 
wealth = pd.DataFrame({"values" : test_file['Total Assets']-test_file['Liabilities']})
print(wealth.mean())
test_file.describe()


# In[206]:


# considering 3 as the threshold for a most wealthy person and 2 as the most criminal type of person. 


# In[207]:


test_file['Party'] = le_party1.inverse_transform(test_file['Party'])
mapping = {}
for index , value in enumerate(parties):
     mapping[value] = index

#iterating through rows
for index, row in test_file.iterrows():
    party_name = row['Party']
    idx = mapping[party_name]
    graph.loc[idx,'Count']+=1
    if(row.loc['Criminal Case'] >= 2):
        graph.loc[idx,'Criminal Case']+=1
    if(wealth.loc[index,'values'] >= 3):
        graph.loc[idx,'Wealth']+=1


# In[226]:


#plotting the graph 
print(graph)
Y1 = 100*(graph['Criminal Case']/graph['Count'])
Y2 = 100*(graph['Wealth']/graph['Count'])
graph['Parties'] = graph['Parties'].replace('Sikkim Krantikari Morcha','SKM')
graph['Parties'] = graph['Parties'].replace('Tipra Motha Party','TMP')
plt.figure(figsize=(15, 6))
plt.scatter(graph['Parties'],Y1)
plt.title('Scatter plot of percentage of most ciminal cases candidates vs parties')
plt.xlabel('Parties')
plt.ylabel('% candidates having most criminal cases')
plt.show()


# In[227]:


plt.figure(figsize=(15, 6))
plt.scatter(graph['Parties'],Y2)
plt.title('Scatter plot of percentage of most wealthy candidates vs parties')
plt.xlabel('Parties')
plt.ylabel('% candidates having most wealth')
plt.show()

