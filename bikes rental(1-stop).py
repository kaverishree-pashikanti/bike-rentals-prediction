#!/usr/bin/env python
# coding: utf-8

# In[144]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


# In[145]:


dataset=pd.read_csv("BoomBikes.csv")
dataset.head()


# In[146]:


dataset.shape


# In[147]:


dataset.columns


# In[148]:


dataset.describe()


# In[149]:


dataset.info()


# In[150]:


#Assigning string values to different seasons instead of numeric values
#1=spring
dataset.loc[(dataset["season"]==1),"season"]="spring"
#2-summer
dataset.loc[(dataset["season"]==2),"season"]="summer"
#3-fall
dataset.loc[(dataset["season"]==3),"season"]="fall"
#4=winter
dataset.loc[(dataset["season"]==4),"season"]="winter"


# In[151]:


dataset["season"].astype("category").value_counts()


# In[152]:


dataset["yr"].astype("category").value_counts()


# In[153]:


def obj_map_mnths(x):
    return x.map({1:"jan",2:"feb",3:"march",4:"april",5:"may",6:"june",7:"july",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"})


# In[154]:


dataset[["mnth"]]=dataset[["mnth"]].apply(obj_map_mnths)


# In[155]:


dataset["mnth"].astype("category").value_counts()


# In[156]:


dataset["holiday"].astype("category").value_counts()


# In[157]:


def str_map_weekday(x):
    return x.map({1:"mon",2:"tues",3:"wed",4:"thurs",5:"fri",6:"sat",0:"sun"})
dataset[["weekday"]]=dataset[["weekday"]].apply(str_map_weekday)
dataset["weekday"].astype("category").value_counts()


# In[158]:


dataset["workingday"].astype("category").value_counts()


# In[159]:


#1=clear,few clouds,partly cloudy
dataset.loc[(dataset["weathersit"]==1),"weathersit"]="A"
#2=mist,cloudly
dataset.loc[(dataset["weathersit"]==2),"weathersit"]="B"
#light snow,heavy rain
dataset.loc[(dataset["weathersit"]==3),"weathersit"]="C"


# In[160]:


dataset["weathersit"].astype("category").value_counts()


# # Data visualization

# In[161]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[162]:


sns.distplot(dataset["temp"])


# In[163]:


sns.distplot(dataset["atemp"])


# In[164]:


sns.distplot(dataset["windspeed"])


# In[165]:


sns.distplot(dataset["cnt"])


# In[166]:


dataset["dteday"]=dataset["dteday"].astype("datetime64")


# In[167]:


dataset_categorical=dataset.select_dtypes(exclude=["float64","datetime64","int64"])
dataset_categorical.columns


# In[168]:


dataset_categorical


# In[169]:


plt.figure(figsize=(20,20))
plt.subplot(3,3,1)
sns.boxplot(x="season",y="cnt",data=dataset)
plt.subplot(3,3,2)
sns.boxplot(x="mnth",y="cnt",data=dataset)
plt.subplot(3,3,3)
sns.boxplot(x="weekday",y="cnt",data=dataset)
plt.subplot(3,3,4)
sns.boxplot(x="weathersit",y="cnt",data=dataset)
plt.subplot(3,3,5)
sns.boxplot(x="workingday",y="cnt",data=dataset)
plt.subplot(3,3,6)
sns.boxplot(x="weekday",y="cnt",data=dataset)
plt.subplot(3,3,7)
sns.boxplot(x="holiday",y="cnt",data=dataset)
plt.show()


# In[170]:


intvarlist=["casual","registered","cnt"]
for var in intvarlist:
    dataset[var]=dataset[var].astype("float")


# In[171]:


dataset_numeric=dataset.select_dtypes("float64")
dataset_numeric.head()


# In[172]:


sns.pairplot(dataset_numeric)
plt.show()


# In[173]:


cor=dataset_numeric.corr()
cor


# In[174]:


mask=np.array(cor)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
fig.set_size_inches(10,10)
sns.heatmap(cor,mask=mask,vmax=0.8,square=True,annot=True)


# In[175]:


dataset.drop('atemp', axis=1,inplace=True)


# In[176]:


dataset.head()


# In[177]:


dataset_categorical=dataset.select_dtypes(include=["object"])


# In[178]:


dataset_categorical.head()


# In[179]:


dataset_dummies=pd.get_dummies(dataset_categorical,drop_first=True)
dataset_dummies.head()


# In[180]:


dataset=dataset.drop(list(dataset_categorical.columns),axis=1)
dataset


# In[181]:


dataset=pd.concat([dataset,dataset_dummies],axis=1)
dataset.head()


# In[182]:


dataset=dataset.drop(["instant","dteday"],axis=1,inplace=False)
dataset


# In[183]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[184]:


from sklearn.model_selection import train_test_split
np.random.seed(0)
dt_train,dt_test=train_test_split(dataset,train_size=0.7,test_size=0.3,random_state=100)
dt_train


# In[185]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[186]:


var=["temp","hum","windspeed","casual","registered","cnt"]
dt_train[var]=scaler.fit_transform(dt_train[var])
dt_train.describe()


# In[187]:


plt.figure(figsize=(30,30))
sns.heatmap(dt_train.corr(),annot=True,cmap="YlGnBu")
plt.show()


# In[188]:


x_train=dt_train.drop(["casual","registered"],axis=1)
y_train=dt_train.pop("cnt")
x_train.head()


# In[189]:


np.array(dt_train)


# In[190]:


import statsmodels.api as sm
x_train_lm=sm.add_constant(x_train)
lr=sm.OLS(y_train,x_train_lm).fit()
lr.params


# In[191]:


x_train_lm


# In[192]:


lm=LinearRegression()
lm.fit(x_train,y_train)
print(lm.coef_)
print(lm.intercept_)


# In[193]:


lr.summary()


# In[194]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[195]:


lm=LinearRegression()
rfe1=RFE(lm)
rfe1.fit(x_train,y_train)
print(rfe1.support_)
print(rfe1.ranking_)


# In[196]:


col1=x_train.columns[rfe1.support_]


# In[197]:


col1


# In[198]:


x_train_rfe1 = x_train[col1]
x_train_rfe1=sm.add_constant(x_train_rfe1)
lm1=sm.OLS(y_train,x_train_rfe1).fit()
lm1.summary()


# In[199]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
a=x_train_rfe1.drop("const",axis=1)

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = a.columns
vif['VIF'] = [variance_inflation_factor(a.values, i) for i in range(a.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[200]:



lm=LinearRegression()
rfe2=RFE(lm)
rfe2.fit(x_train,y_train)
print(rfe1.support_)
print(rfe1.ranking_)


# In[201]:


col2=x_train.columns[rfe2.support_]
x_train_rfe2=x_train[col2]
x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
lm2.summary()


# In[202]:


b=x_train_rfe2.drop("const",axis=1)

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif1 = pd.DataFrame()
vif1['Features'] = b.columns
vif1['VIF'] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif1['VIF'] = round(vif['VIF'], 2)
vif1 = vif.sort_values(by = "VIF", ascending = False)
vif1


# In[203]:


y_train_cnt=lm2.predict(x_train_rfe2)
fig=plt.figure()
sns.distplot((y_train,y_train_cnt),bins=20)


# In[204]:


dt_test[var]=scaler.transform(dt_test[var])
dt_test


# In[205]:


y_test=dt_test.pop("cnt")
x_test=dt_test.drop(["casual","registered"],axis=1)
x_test.head()


# In[206]:


c=x_train_rfe2.drop("const",axis=1)


# In[207]:


col2=c.columns


# In[208]:


x_test_rfe2=x_train[col2]


# In[209]:


x_test_rfe2=sm.add_constant(x_test_rfe2)
x_test_rfe2.info()

