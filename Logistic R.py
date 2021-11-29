#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


#load dataset
dataFrame = pd.read_csv("Iris.csv")
dataFrame.head(5)


# In[3]:


dataFrame.describe()


# In[4]:


#check the data if there's a NaN value
dataFrame.isna().values.any()

#check the features
print(dataFrame.dtypes)


# In[6]:


#prepare the training set
X = dataFrame.iloc[:, :-1]
y = dataFrame.iloc[:, -1]
#plot the relation of each features to the target
plt.xlabel("Features")
plt.ylabel("Species")

pltX = dataFrame.loc[:, "SepalLengthCm"]
pltY = dataFrame.loc[:, "Species"]
plt.scatter(pltX, pltY, color="blue", label="SepalLengthCm")

pltX = dataFrame.loc[:, "SepalWidthCm"]
pltY = dataFrame.loc[:, "Species"]
plt.scatter(pltX, pltY, color="green", label="SepalWidthCm")

pltX = dataFrame.loc[:, "PetalLengthCm"]
pltY = dataFrame.loc[:, "Species"]
plt.scatter(pltX, pltY, color="red", label="PetalLengthCm")

pltX = dataFrame.loc[:, "PetalWidthCm"]
pltY = dataFrame.loc[:, "Species"]
plt.scatter(pltX, pltY, color="black", label="PetalWidthCm")


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)
print(y_pred)


# In[9]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[10]:


import seaborn as sns
import numpy as np
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
Text(0.5, 257.44, 'Predicted label')


# In[11]:


#check precision, recall, f1-score
print(classification_report(y_test, y_pred))
print("accuracy: ", accuracy_score(y_test, y_pred))


# In[ ]:




