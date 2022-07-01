#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[4]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.datasets import load_boston   #sample data in sklearn


# In[7]:


boston=load_boston()


# In[8]:


boston.keys()


# In[9]:


boston.data


# In[11]:


boston.target


# In[12]:


boston.feature_names


# In[13]:


boston.DESCR


# In[16]:


bos=pd.DataFrame(boston.data,columns=boston.feature_names)
bos


# In[17]:


bos["target"]=boston.target


# In[18]:


bos.columns


# In[19]:


bos


# In[ ]:


df=pd.read_csv("Filename.csv")
df.head()


# In[ ]:


df=pd.read_csv("Filename.csv",sep=";",na_values=".")
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.columns


# In[ ]:


df.isnull()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info


# # Summary statistics

# In[ ]:


df.describe()


# EDA

# In[ ]:


df.columnname.unique()


# In[ ]:


df.columnname.value_counts()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


dfcor=df.corr()


# In[ ]:


sns.heatmap(dfcor)


# In[3]:


get_ipython().run_line_magic('pinfo', 'sns.color_palette')


# In[ ]:


plt.figure(figsize=(6,4))
sns.heatmap(dfcor,cmap="Blues",annot=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(dfcor,cmap="YlorRd",annot=True)


# # plotting outliers

# In[ ]:


df.columns


# In[ ]:


sns.countplot(x="Column Name",data=df)


# In[ ]:


df["ColumnName"].plot.box()


# In[ ]:


df.shape


# In[ ]:


collist=df.columns.values
ncol=12
nrows=10


# In[ ]:


plt.fgure(figsize=(ncol,5*ncol))
for i in range(1,len(collist)):
    plt.subplot(nrows,ncol,i+1)   #subplot means box plot is made side by side
    sns.boxplot(df[collist[i]],color"green,orient="v")
    plt.tight_layout()


# # To check the distribution of skewness

# In[ ]:


sns.distplot(df["Column name"])


# In[ ]:


sns.distplot(df["Column name"],kde=False,bins=20)


# In[ ]:


plt.fgure(figsize=(16.16))
for i in range(0,len(collist)):
    plt.subplot(nrows,ncol,i+1)   #subplot means box plot is made side by side
    sns.distplot(df[collist[i]])


# In[ ]:


plt.scatter(df["Column name"],df["Column name"])


# In[ ]:


sns.pairplot(df)


# # Droping a column

# In[ ]:


df.drop("Column name",axis=1,inplace=True )


# In[ ]:


df.dropna(inplace=True)


# # Replacing NaN values with mean

# In[ ]:


df["Column Name"].replace(np.NaN,df["Column Name"].mean,inplace=True)


# # Removing outliers

# In[ ]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[ ]:


threshold=3
print(np.where(z>3))


# In[ ]:


df_new=df[(z<3).all(axis=1)]   #taking only values whose z score is less than 3


# In[ ]:


df.shape


# In[ ]:


df_new.shape


# In[ ]:


df.skew()


# # Transformation

# In[ ]:


df["Column name"]=np.log(df["Column name"])
df["Column name"].plot.hist()


# In[ ]:


from scipy.stats import boxcox
# 0 is for log transform
# .5 is for square root transform
df["column name"]=boxcox(bos["Column name"],0)


# # Splitting  the data into traning and test data sets

# In[ ]:


x=df.iloc[:,0:-1]
x.head()


# In[ ]:


y= df.iloc[:,-1]
y.head()


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# # Linear Regression

# In[20]:


lm=LinearRegression()


# In[ ]:


lm.fit(x_train,y_train)    #fit means traning


# In[ ]:


lm.coef_


# In[ ]:


lm.intercept_


# In[ ]:


lm.score(x_train,y_train)


# In[ ]:


pred=lm.predict(x_test)
print("Predicted result price:",pred)
print("actual price",y_test)


# In[ ]:


print("error:")

print("Mean absolute error:",mean_absolute_error(y_test,pred))
print("Mean squared error:",mean_squared_error(y_test,pred))

print("Root Mean squared error:"np.sqrt(Mean_squared_error(y_test,pred)))


# In[ ]:


#r2 score is the coefficient of determination
#i.e, chamge comin in y whenever x is being changed

from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# # To save the data to csv

# In[ ]:


df=pd.DataFrame(pred)
df.to_csv("File name to be saved.csv")


# # Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)   # change the random state values to get good accuracy score


# In[ ]:


lg=LogisticRegression


# In[ ]:


lg.fit(train_x,train_y)


# In[ ]:


pred=lg.predict(test_x)
print(pred)


# In[ ]:


# print("accuracy_score:",accuracy_score(test_y,pred))


# In[ ]:


print(confusion _matrix(test_y,pred))


# In[ ]:


print(classification_report(test_y,pred))


# # Classification

# In[24]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# # MultinomialNB

# In[ ]:


mnb=MultinomialNB()       #GucissionNB is used only for 2 output and MNB is for more than 2 outputs
mnb.fit(x_train,y_train)
predmnb=mnb.predict(x_test)
print(accuracy_score(y_test,predmnb))
print(confusion_matrix(y_test,predmnb))
print(classification_report(y_test,predmnb))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
predgnb=gnb.predict(x_test)
print(accuracy_score(predgnb,y_test))
print(confusion_matrix(y_test,predgnb))
print(classification_report(y_test,predgnb))


# # Support Vector classifier

# In[ ]:


svc=SVC(kernel="rbf ")    #rbf is default
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# # Decision Tree Clasifier

# In[ ]:


dtc=DecisionClassifier()    
dtc.fit(x_train,y_train)
dtc.score(x_train,y_train)
preddtc=dtc.predict(x_test)
print(accuracy_score(y_test,preddtc))
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# # KNeighbors Clasifier

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)    
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
predknn=knn.predict(x_test)
print(accuracy_score(y_test,predknn))
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))


# # SVC with different kernel

# In[ ]:


svc=SVC(kernel="poly ")   
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# In[ ]:


def svmkernel(ker):
    svc=SVC(kernel="poly ")   
    svc.fit(x_train,y_train)
    svc.score(x_train,y_train)
    predsvc=svc.predict(x_test)
    print(accuracy_score(y_test,predsvc))
    print(confusion_matrix(y_test,predsvc))
    print(classification_report(y_test,predsvc))


# In[ ]:


svmkernel("rbf")


# In[ ]:


svmkernel("poly")


# # Save the best algorithm

# In[ ]:


df=pd.DataFrame(predsvc)
df.to.csv("svc_prediction.csv")


# In[ ]:


model=[DecisionTreeClassifier(),SVC(),KNeighborsClassifier(),MultinomialNB()]


# In[ ]:


for m in model:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    predm=m.predict(x_test)
    print(accuracy_score(y_test,predm))
    print(confusion_matrix(y_test,predm))
    print(classification_report(y_test,predm))
    print("\n")


# # Regularization:: L1 & L2 Regularization

# In[25]:


from sklearn.linear_model import Lasso,Ridge


# # L1 (Lasso)

# In[ ]:


#will reduce the coefficient to zero
ls=Lasso(alpha=0.0001)   #alpha values could be 0.0001,0.001,0.01,0.1,1,10.... Higher the values reduces all coeffients towards 0 and impact output
ls.fit(x_train,y_train)   # default value of alpha =1.0
ls.score(x_train,y_train)


# In[ ]:


ls.coef_


# In[ ]:


plt.bar(File name.features_names,ls.coef_)
plt.show()


# # L2(Ridge)

# In[ ]:


#try to minimize the coefficient variance

rd=Ridge(alpha=0.001)
rd.fit(x_train,y_train)
rd.score(x_train,y_train)


# In[ ]:


rd.coef_


# In[ ]:


plt.bar(File name.features_names,rd.coef_)
plt.show()


# # Elastic Net

# In[ ]:


#Elastic Net is a combination of both lasso and ridge

from sklearn.linner_model import ElasticNet
enr=ElasticNet(alpha=0.0001)
enr.fit(x_train,y_train)
predenr=enr.predict(x_test)
print(enr.score(x_train,y_train))
enr.coef_


# # SVR,DTR,KNNR for regression

# In[27]:


from sklearn.svm import SVR


# In[ ]:


svr=SVR(kernel="poly")   
svr.fit(x_train,y_train)
svr.score(x_train,y_train)
predsvr=svr.predict(x_test)


# In[ ]:


svr=SVR(kernel="Linear")   
svr.fit(x_train,y_train)
svr.score(x_train,y_train)
predsvr=svr.predict(x_test)


# In[ ]:


svr=SVR(kernel="rbf")   
svr.fit(x_train,y_train)
svr.score(x_train,y_train)
predsvr=svr.predict(x_test)


# In[ ]:


kernellist=["linear","poly",rbf]
for i in kernellist:
    sv=SVR(kernel=i)
    sv.fit(x_train,y_train)
    print(sv.score(x_train,y_train))


# # Cross Validation

# In[29]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnbscores=cross_val_score(gnb,x,y,cv=5)
print(gnbscores)
print(gnbscores.mean(),gnbscores.std())


# In[ ]:


svcscores=cross_val_score(svc,x,y,cv=5)   #mention svc as mentioned which cheacking SVC
print(svcscores)
print(svcscores.mean(),svcscores.std())


# In[ ]:


dtcscores=cross_val_score(dtc,x,y,cv=5)
print(dtcscores)
print(dtcscores.mean(),dtcscores.std())


# In[ ]:


knnscores=cross_val_score(knn,x,y,cv=5)
print(knnscores)
print(knnscores.mean(),knnscores.std())


# # Saving the best model(Serialization)

# In[30]:


import joblib


# In[ ]:


joblib.dump(dtc,"dtcfile.obj")   #load the best model from the file


# In[ ]:


dtc_from_joblib=joblib.load("dtcfile.obj")   #Used to retrive the obj file

dtc_from_joblib.predict(x_test)  #use the loaded model to make predictions


# Second method

# In[ ]:


import pickle
filename="pickledtcfile.pkl"
pickle.dump(dtc,open(filename,"wb"))

#load the model from disk
loaded_model=pickle.load(open(filename,"rb"))

loaded_model.predict(x_test)


# # AUC-ROC Curve

# In[31]:


from sklearn.metrics import roc_curve


# In[32]:


import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


# In[ ]:


y_pred_prob=lg.predict_proba(x_test)[:,1]  #predict the probility of 1


# In[ ]:


y_pred_prob


# In[ ]:


fpr,,tpr,thresholds=roc_curve(y_test,y_pred_prob)


# In[ ]:


fpr


# In[ ]:


tpr


# In[ ]:


thresholds


# In[ ]:


plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Logistic Regression")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Logistic Regression")
plt.show()


# In[ ]:


auc_score=roc_auc_score(y_test,lg.predict(y_test))
#print(auc_score)


# In[ ]:


#Decision tree Curve


y_pred_prob=dtc.predict_proba(x_test)[:,1]      #for decision tree classifier
fpr,,tpr,thresholds=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="Decision tree clasifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Decision tree clasifier")
plt.show()
auc_score=roc_auc_score(y_test,dtc.predict(y_test))
print(auc_score)


# In[ ]:


# KNN curve

y_pred_prob=knn.predict_proba(x_test)[:,1]      
fpr,,tpr,thresholds=roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="KNeighbor clasifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("KNeighbor clasifier")
plt.show()
auc_score=roc_auc_score(y_test,knn.predict(y_test))
print(auc_score)


# # Simple Imputer

# In[35]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# with most frequent data

# In[ ]:


imp=SimpleImputer(strategy="most_frequent")  #only for string values

df["column name"]=imp.fit_transform(df["column name"].values.reshape(-1,1))

df


# With Mean

# In[ ]:


imp=SimpleImputer(missing_values=np.nan,strategy="mean")

df["Column name"]=imp.fit_transform(df["column name"].values.reshape(-1,1))

#imputer works for numerical data such as age and salary


# # Label Encoder

# In[ ]:


le=LabelEncoder()
df["Column name"]=le.fit_transform(df["Column name"])

#for more than 1 column
lilist=["Column name1","column name2"]
for val in list:
    df[val]=le.fit_transform(df[val].astype(str))


# other ways in replace method

# In[ ]:


df["column name"].replace(9,numpy.NaN,inplace=True)


# In[ ]:


from sklearn.preprocessing import Imputer
imp= Imputer(missing_values="NaN",strategy="most_frequent")
df["column name"]=imp.fit_transform(df["column name"].values.reshape(-1,1))


# For column

# In[ ]:


df["column name"]=df["column name"].replace(np.nan,0)


# In[ ]:


df["age"]=df["age"].replace(np.nan,df["age"].mean())


# for whole dataframe

# In[ ]:


df=df.replace(np.nan,0)


df=df.replace(np.nan,df.mean())


# In[ ]:


df.replace(-1,np.nan,inplace=True)     # replace -1 with nan values

df.replace(-1,df.mean(),inplace=True)  


# # PCA

# In[36]:


from sklearn.decomposition import PCA


# In[ ]:


x=df.iloc[:,0:-1]
x.shape


# In[ ]:


pca=PCA(n_components=10)


# In[ ]:


y=df.iloc[:,-1]


# In[ ]:


x=pca.fit_transform(x)
x.shape


# In[ ]:


pd.DataFrame(df=x)


# # Scaling

# standard scaler

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scale=StandardScaler()


# In[ ]:


x=scale.fit_transform(x)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)  


# # Ensemble Methods

# # Random Forest Classifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
predrf=rf.predict(x_test)
print(accuracy_score(y_test,predrf))
print(confusion_matrix(y_test,predrf))
print(classification_report(y_test,predrf))


# # AdaBoost Classifier

# In[40]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


#AdaBoostClassifier(base_estimator=Decision TreeClassifier(),n_estimators=50,learning_rate=1.0)
ad=AdaBoostClassifier()
ad.fit(x_train,y_train)
predad=ad.predict(x_test)
print(accuracy_score(y_test,predad))
print(confusion_matrix(y_test,predad))
print(classification_report(y_test,predad))


# In[ ]:


ad=AdaBoostClassifier(n_estimators=50)
ad.fit(x_train,y_train)
predad=ad.predict(x_test)
print(accuracy_score(y_test,predad))
print(confusion_matrix(y_test,predad))
print(classification_report(y_test,predad))


# In[ ]:


svc=SVC()

ad=AdaBoostClassifier(n_estimators=50,base_estimator=svc,algorithm="SAMME")   #algoruth SAMME.R is default
ad.fit(x_train,y_train)


predad=ad.predict(x_test)
print(accuracy_score(y_test,predad))
print(confusion_matrix(y_test,predad))
print(classification_report(y_test,predad))


# In[ ]:


svc=SVC(probabitity=True,kernel="linear") #for ablove kernel="rbf" is default.


# Random forest and adaboost can be used for regression also

# # Hyper parameter tuning(Grid serach CV)

# In[42]:


from sklearn import svm,datasets
from sklearn.model_selection import GridSearchCV


# In[43]:


iris=datasets.load_iris()


# In[44]:


parameters={"kernel":["linear","rbf"],"C":[1,10]}
svc=svm.SVC()
clf=GridSearchCV(svc,parameters)
clf.fit(iris.data,iris.target)

clf


# In[45]:


print(clf.best_params_)


# In[46]:


sv=svm.SVC(kernel="linear",C=1)
sv.fit(iris.data,iris.target)
t=sv.score(iris.data,iris.target)
print(round(t,2))


# with DTC

# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[49]:


dtc=DecisionTreeClassifier()
iris=datasets.load_iris()


# In[51]:


grid_param={"criterion":["gini","entropy"]}


# In[53]:


gd_sr=GridSearchCV(estimator=dtc,param_grid=grid_param,scoring="accuracy",cv=5)

gd_sr.fit(iris.data,iris.target)

best_parameters=gd_sr.best_params_
print(best_parameters)
best_results = gd_sr.best_score_
print(best_results)


# In[54]:


dtc=DecisionTreeClassifier(criterion="gini")
dtc.fit(iris.data,iris.target)
dtc.score(iris.data,iris.target)


# # PipeLine

# In[55]:


#create a pipeline that standardizes the data then create model

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


array=dataframe.values


# In[ ]:


x=array[:,0:8]


# In[ ]:


y=array[:,8]


# In[ ]:


#create the pipeline

estimators=[]
estimators.append(("standardize",StandardSCaler()))
estimators.append(("dtc",DecisionTreeClassifier()))
model=pipeline(estimators)


# In[ ]:


#evaluate pipeline

kfold=KFold(n_spilts=10,random_state=8)
results=cross_val_score(model,x,y,cv=kfold)
print(results.mean())


# # Unsupervised Data

# In[57]:


from sklearn.datasets import load_iris
iris=load_iris()
print(iris.data)


# In[58]:


print(iris.target)


# In[59]:


from sklearn.cluster import KMeans


# In[61]:


kmeans=KMeans(n_clusters=3)
kmodel=kmeans.fit(iris.data)


# In[62]:


kmodel.cluster_centers_


# In[64]:


import pandas as pd
pd.crosstab(iris.target,kmodel.labels_)


# In[65]:


from sklearn.metrics import homogeneity_score
print(homogeneity_score(kmodel.labels_,iris.target))


# # Steps to be followed

# In[66]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#import read csv file


# In[ ]:


df=pd.DtaFrame(data=df)


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.info()


# In[ ]:


df.isnull.sum()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df.describe()


# In[ ]:


fig=plt.fiqure(figsize=(10,5))
hc=df.corr(method="pea rson")
sns.heatmap(hc,annot=True,cmap="Blues")


# In[ ]:


sns.countplot(x="class",data=df)
plt.show()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.skew()


# In[ ]:


EDA(do all the graph)


# In[ ]:


#to remove skewness is data

df["column name"]=np.log(df["column name"])
sns.histplot(df["column name"])


df["column name"].skew()


# In[ ]:


df["column name"]=np.sqrt(df["column name"])
sns.histplot(df["column name"])

df["column name"].skew()


# In[ ]:


"yeo-johnson" works with "+ve" and "-ve" values
box-cox works only with strictly "+ve" values

pt=PowerTransformer(method="yeo-johnson",standardize=True)
c=["column names"]
df["column name"]=pt.fit_transform(df["column name"])


# In[ ]:


zscr=np.abs(zscore(df))

threshold=3
print(np.where(zscr>3))


# In[ ]:


dfnew=df(zscr<3).all(axis=1)

print("shape after removing the outliers:"dfnew.shape)


# In[ ]:


df=dfnew


# In[ ]:


df_x=df.drop(columns=["column name"])  #if skewness is more drop the column
y=df[["column name"]]


# In[ ]:


for i in df_x.columns:
    if df_x[i].skew()>0.5:
        df_x[i]np.cbrt(df_x[i])
    if df_x[i].skew()<-0.5:
        df_x[i]=np.cbrt(df_x[i])
df_x.skew()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(df_x)

x=pd.DataFrame(x,columns=df_x.columns)
x           


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


create the train test split


# In[ ]:


use all the algorithms


# In[ ]:


Cross validation


# In[ ]:


joblib


# In[ ]:


model=[lg,sv,gnb,rf,ad]

for m in model:
    m.fit(x_train,y_train)
    pred=m.predict(x_test)
    print("accuracy score: of",m)
    print(accuracy_score(y_test,pred))
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    score=cross_val_score(m,x,y,cv=5)
    print(score)
    print(score.mean)

