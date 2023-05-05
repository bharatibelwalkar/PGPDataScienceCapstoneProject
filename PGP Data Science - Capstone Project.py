#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Importing essential libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


###Importing data file from my computer 
df = pd.read_csv("D:\AIR 2020-2022\Data Science\PGP Data Science Program\Module 9 -- Capstone\Project 2\Healthcare - Diabetes\health care diabetes.csv")


# In[3]:


###Checking out the data structure 
df.head(10)


# In[4]:


###Confirming the shape of the data file (number of rows = patients AND number of columns = variables)
df.shape


# In[5]:


###Question 1: Exploring dataset with descriptive analyses  
df.describe()


# In[6]:


###Question 1: Transposing the results for userfrieldliness
df.describe().T


# In[7]:


###Question 1: Find the number of missing values for the following list of variables:""
##•	Glucose
##•	BloodPressure
##•	SkinThickness
##•	Insulin
##•	BMI

df['Glucose'].value_counts()[0]


# In[8]:


df['BloodPressure'].value_counts()[0]


# In[9]:


df['SkinThickness'].value_counts()[0]


# In[10]:


df['Insulin'].value_counts()[0]


# In[11]:


df['BMI'].value_counts()[0]


# In[27]:


##OR anothee way to find "0" values for multiple variables at once
df[df[["Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI"]]==0].count()


# In[30]:


##Calculating % of missing values in the dataset (by dividing with the lenght-# of rows in the datset)
df[df[["Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI"]]==0].count()/len(df)*100


# In[12]:


##Question 3: count frequency plot 
df.info()


# In[13]:


df_box = df.drop('Outcome',axis=1)


# In[14]:


fig , ax = plt.subplots(nrows= 2,       # no,of plots comes in row wise 
                        ncols= 4,       # no,of plots comes in column wise 
                        figsize=(20,10) # size of plot
                        )
ax = ax.flatten() # It returns a flattened version of the array, to avoid numpy.ndarray
index = 0
for i in df_box.columns:
  sns.boxplot(y=i,data = df_box, ax=ax[index])
  index += 1
plt.tight_layout(pad=0.4)


# In[29]:


##Question 2: Histograms 

df.hist(bins=15)    

plt.tight_layout(rect=(0, 0, 2, 2)) # it will change the size of the plot

plt.suptitle('Question 2 Histograms',
             x=1, # title x position
             y=2, # title y position
             fontsize=14) 


# In[16]:


##Question 5: Scatter chart for pairs of variables
sns.pairplot(df, hue='Outcome')


# In[19]:


###Question 6: correlation analysis with heatmap

corr = df.corr()
sns.heatmap(corr,
            fmt='.1f',
            annot = True)


# In[21]:


df.corr()


# In[31]:


df.unique():


# In[35]:


###Question 2: Replacing the missing values in specified variables with their respective medians
##Calculating both medians = with and without including zero values
##Going for median replacement. If I had done outlier treatment, I could have gone with means

for i in ["Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI"]:
    print(i, "old median:", df[i].median())
    median_value=df[df[i]!=0][i].median()
    print(median_value,"\n")
    df[i].replace(0,median_value,inplace=True)


# In[36]:


##After median replacement, confirming whether the vairables inlcude any missing values still?
df[df[["Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI"]]==0].count()


# In[38]:


##Question 4: checking the balance of data by plotting the count of outcome
df.Outcome.value_counts(normalize=True)*100


# In[39]:


###Question 4: plotting the outcome of the data
sns.countplot(df.Outcome)


# In[40]:


###Data Modeling (Questions 7 through 10)
df.columns


# In[41]:


x = df.drop("Outcome", axis=1) #Predictor Variables
y = df["Outcome"] #Target Variable


# In[43]:


df["Outcome"].value_counts(normalize=True)


# In[52]:


##Stratify command is used because of the imabalnced outcome variable values in the data
##This commas will preserve the ratio of the dependent variable in train/test by creating homoeneous groups
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.20, random_state=42, stratify=y)


# In[45]:


train_x.shape


# In[46]:


test_x.shape


# In[50]:


train_y.value_counts(normalize=True)


# In[53]:


test_y.value_counts(normalize=True)


# In[55]:


###Question 8: K-Nearest Neighbor algorithm is used for this problem
##Scaling is necessary for this algorithm especially given that all predictor variables are on different scale

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[58]:


scaler=StandardScaler()


# In[64]:


scalled_train_x = scaler.fit_transform(train_x)


# In[65]:


scalled_test_x = scaler.transform(test_x)


# In[68]:


Belwalkar_captone_model = KNeighborsClassifier(n_neighbors=5)


# In[70]:


Belwalkar_captone_model.fit(scalled_train_x,train_y)


# In[71]:


Belwalkar_captone_model.score(scalled_train_x, train_y)


# In[72]:


Belwalkar_captone_model.score(scalled_test_x, test_y)


# In[73]:


from sklearn.metrics import classification_report,confusion_matrix


# In[74]:


y_pred = Belwalkar_captone_model.predict(scalled_test_x)


# In[75]:


pd.DataFrame(y_pred).value_counts()


# In[76]:


print(classification_report(test_y,y_pred))


# In[77]:


print(confusion_matrix(test_y,y_pred))


# In[83]:


acc=[]
ran=range(2,15)
for k in ran:
    Belwalkar_captone_model = KNeighborsClassifier(n_neighbors=k)
    Belwalkar_captone_model.fit(scalled_train_x,train_y)
    test_acc= Belwalkar_captone_model.score(scalled_test_x,test_y)
    
    acc.append(test_acc*100)

sns.lineplot(ran,acc)


# In[85]:


###Alternate model of Random Forest

from sklearn.ensemble import RandomForestClassifier
Belwalkar_captone_RF_model =RandomForestClassifier(n_estimators=500,max_depth=3)


# In[87]:


Belwalkar_captone_RF_model.fit(scalled_train_x,train_y)


# In[88]:


Belwalkar_captone_RF_model.score(scalled_train_x,train_y)


# In[90]:


Belwalkar_captone_RF_model.score(scalled_test_x,test_y)


# In[93]:


RF_y_pred=Belwalkar_captone_RF_model.predict(scalled_test_x)


# In[94]:


pd.DataFrame(RF_y_pred).value_counts()


# In[95]:


print(classification_report(test_y,RF_y_pred))


# In[96]:


print(confusion_matrix(test_y,RF_y_pred))


# In[131]:


def plot_roc_curve(test_y,RF_y_pred):
    fpr,tpr,thresholds = roc_curve(test_y,RF_y_pred)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.show()


# In[132]:


plot_roc_curve(test_y,RF_y_pred)
print(f'model_1_AUC_score: {roc_auc_score(test_y,RF_y_pred)}')


# In[135]:


import numpy as np
from sklearn import metrics


# In[138]:


fpr, tpr, thresholds = metrics.roc_curve(test_y,RF_y_pred, pos_label=2)


# In[139]:


fpr, tpr, thershold = roc_curve(y_test, rfc1.predict_proba(x_test)[:,1])
rfc_roc = roc_auc_score(y_pred_rfc1,y_test)
plt.figure()
plt.subplots(figsize=(15,10))
plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)'%rfc_roc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0,1.0])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate (1-specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('Receiver operating characteristic for Random Forest Classifier ')
plt.legend(loc ="lower right")
plt.show()


# In[140]:


plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[ ]:




