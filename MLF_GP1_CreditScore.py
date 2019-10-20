#!/usr/bin/env python
# coding: utf-8

# In[213]:


# Reference 1 : Codes & Hints provided by Prof. Matthew Murphy
# Reference 2 : Raschka textbook

# importing relevant libraries

import sklearn
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns


# In[214]:


# Loading credit score data

cs_df=pd.read_csv('MLF_GP1_CreditScore.csv')


# In[186]:


# setting larger column limit for cell output

pd.options.display.max_columns = 2000


# In[187]:


# printing top 5 samples of dataset

cs_df.head()


# In[188]:


# summary statistics of entire dataset

cs_df.describe()


# In[189]:


# checking for missing values

cs_df.isnull().values.sum()


# In[190]:


# print data types of all the variables

cs_df.dtypes


# In[191]:


# print info of the dataset

print(cs_df.info())


# In[192]:


# counts of class variables

count_Inv = cs_df['InvGrd'].value_counts()

count_rating = cs_df['Rating'].value_counts()

print(count,"\n")
print(count_rating)


# In[193]:


# making a copy of the dataset

cs_df_rp = cs_df


# In[158]:


# checking the copy

print(cs_df_rp.shape, "/n")
print(cs_df_rp.head())


# In[194]:


# creating dictionary to map rating values to numeric values

labels = cs_df_rp['Rating'].astype('category').cat.categories.tolist()

replace_map_comp = {'Rating' : {k : v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print (replace_map_comp)


# In[195]:


# replacing values of Rating with the numeric values

cs_df_rp.replace(replace_map_comp, inplace = True)

print(cs_df_rp.head())


# In[196]:


# printing counts of rating variable in the new dataset, after replacement

cs_df_rp['Rating'].value_counts()


# In[215]:


cs_df.head()


# In[197]:


# creating feature matrix

X = cs_df_rp.drop(['InvGrd','Rating'], axis = 1)


# In[198]:


# checking dimension of X

print (X.shape,"\n")


# In[199]:


# printing top 5 samples of X

X.head()


# In[200]:


# printing bottom 5 samples of X

X.tail()


# In[201]:


# creating dependent variable vector, for InvGrd

y_i = cs_df_rp['InvGrd']


# In[202]:


# checking dimension of y_i

y_i.shape


# In[203]:


# printing samples of y_i

print(y_i.head())
print(y_i.tail())


# In[204]:


# creating dependent variable vector, for rating

y_r = cs_df_rp['Rating']


# In[205]:


# checking dimension of y_r

y_r.shape


# In[206]:


# printing samples of y_r

print(y_r.head())
print(y_r.tail())


# In[172]:


# Box plot of the dataset

plt.figure(figsize=(10,10))

plt.xticks(rotation=90)

sns.boxplot(data = cs_df_rp)


# In[207]:


# There are some outliers and imputations can be done to improve the model performance


# In[ ]:





# In[220]:


# Plotting Histogram of Investment Grade

hist = plt.figure(figsize =(10,10))
hist = plt.xlabel('Investment Grade')
hist = plt.ylabel('Frequency')

pd.Series(cs_df['InvGrd']).value_counts().plot('bar')


# In[209]:


# Printing counts of InvGrd variable

pd.Series(cs_df['InvGrd']).value_counts()


# In[221]:


# Plotting Histogram of Rating

hist = plt.figure(figsize =(10,10))

hist = plt.xlabel('Rating')
hist = plt.ylabel('Frequency')

pd.Series(cs_df['Rating']).value_counts().plot('bar')


# In[217]:


# printing counts of Rating

pd.Series(cs_df['Rating']).value_counts()


# In[222]:


# creating index of column names in the dataset

a=cs_df_rp.columns


# In[223]:


# calculating correlation matrix and plotting heatmap

corr = cs_df_rp.corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,yticklabels=a,
                      xticklabels=a)


# In[112]:


# variables for scatter plots (taking variables which are correlated and few uncorrelated variables)

corr_vars_1 = ['Sales/Revenues','Current Liabilities','EBITDA','EBITDA Margin','Net Income Before Extras',
             'EPS Before Extras','ROA','ROE','Current Liquidity','Cash','InvGrd','Rating']


corr_vars_2 = ['Total Debt','LT Debt','Total Debt/EBITDA','Total Debt/MV','Net Debt/EBITDA','Net Debt/MV', 
               'Net Debt','InvGrd','Rating']


# In[113]:


# printing correlation matrix

print(corr)


# In[114]:


# scatter plots of correlated variables - 1

plot1 = plt.figure(figsize=(20,20))

plot1 = sns.pairplot(cs_df_rp[corr_vars_1],height=2)
plot1 = plt.tight_layout()
plt.show()


# In[115]:


# scatter plots of correlated variables - 2

plot2 = plt.figure(figsize=(20,20))

plot2 = sns.pairplot(cs_df_rp[corr_vars_2],height=2)
plot2 = plt.tight_layout()
plt.show()


# In[116]:


# Feature selection using correlation heatmap

# Variables that can be removed because of correlation for feature selection

# from scatter plots 1 -  ('EBITDA', 'EBITDA Margin'), ('Net Income Before Extras', 'ROA', 'ROE'), 
# ('Current Liquidity', 'Cash')
# these combinations are correlated, so out of these 7 variables, 3 variables can be selected :
# selection based on correlation with 'InvGrd' & 'Rating' : 
# selected ones for 'InvGrd' : ('EBITDA','Net Income Before Extras', 'Current Liquidity')
# selected ones for 'Rating' : ('EBITDA', 'ROE', 'Current Liquidity')

# from scatter plots 2 - ('Total Debt','LT Debt','Total Debt/EBITDA','Total Debt/MV')
# ('Net Debt/EBITDA','Net Debt/MV', 'Net Debt')
# these combinations are correlated, so out of these 7 variables, 2 variables can be selected :
# selected ones for 'InvGrd' : ('Total Debt','Net Debt')
# selected ones for 'Rating' : ('Total Debt/MV','Net Debt/MV')

# So from these 14 variables, we can select 5 variables for our model, without loss of explained variance
# (we have to check using evaluation metrics, whether this feature selection gives best results


# In[ ]:


# Binary Classification


# In[225]:


# PCA

from sklearn.decomposition import PCA

pca = PCA()

X_pca = pca.fit_transform(X)

exp_var = pca.explained_variance_ratio_

cum_exp_var = np.cumsum(exp_var)

print("explained variance: ", exp_var, "\n")
print("cumulative explained variance: ", cum_exp_var)


# In[226]:


len(exp_var)


# In[227]:


# Plotting explained variance

pca_ev = plt.bar(range(1,27),exp_var,
                label = 'Individual explained variance')
pca_ev = plt.step(range(1,27),cum_exp_var,
                 label = 'Cumulative explained variance')

pca_ev = plt.ylabel("Explained variance ratio")
pca_ev = plt.xlabel("Principal component index")
pca_ev = plt.legend(loc = 'best')
plt.show()


# In[228]:


# PCA with 13 components

pca_13 = PCA(n_components = 13)

X_pca_13 = pca_13.fit_transform(X)


# In[229]:


# Checking dimension of PCA dataset

X_pca_13.shape


# In[232]:


# Printing samples of PCA dataset

X_pca_13[0:3]


# In[235]:


import time


# In[234]:


# Splitting dataset into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_i_train,y_i_test  =  train_test_split(X,y_i,test_size = 0.1, stratify = y_i, random_state = 42)

X_pca_train,X_pca_test,y_i_pca_train,y_i_pca_test = train_test_split(X_pca_13,y_i,test_size = 0.1, stratify = y_i, random_state = 42)


# In[237]:


# Checking Dimensions

print("X_train dimension: ", X_train.shape)
print("X_test dimension: ", X_test.shape)
print("y_i_train dimension: ", y_i_train.shape)
print("y_i_test dimension: ", y_i_test.shape)

print("X_pca_train dimension: ", X_pca_train.shape)
print("X_pca_test dimension: ", X_pca_test.shape)
print("y_i_pca_train dimension: ", y_i_pca_train.shape)
print("y_i_pca_test dimension: ", y_i_pca_test.shape)


# In[261]:


# Logistic Regression - basic

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

logit      = LogisticRegression(multi_class = 'auto')

solver     = ['liblinear']

penalty    = ['l1','l2']

c          = [100,10,1,0.1,0.01,0.001]

max_iter   = [100,500,1000]

param_grid = [{'penalty' : penalty,'C': c,'max_iter': max_iter,'solver': solver}]


# In[288]:


# Model Selection by hyperparameter tuning

t_lr_s     = time.time() 

gs = GridSearchCV(logit,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs.fit(X_train,y_i_train)

t_lr_e     = time.time()


print("time taken for logit with all attributes: ",(t_lr_e - t_lr_s))


# In[263]:


gs.best_score_


# In[264]:


gs.best_params_


# In[266]:


lr = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 100, multi_class = 'auto')


# In[267]:


lr.fit(X_train,y_i_train)


# In[271]:


lr.intercept_


# In[274]:


lr.coef_


# In[284]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


# In[270]:


y_i_pred_train_lr = lr.predict(X_train)
y_i_pred_test_lr  = lr.predict(X_test)


# In[277]:


print(y_i_pred_train_lr[0:100])
print(y_i_pred_test_lr[0:100])


# In[278]:


print("training accuracy score: ", accuracy_score(y_i_train,y_i_pred_train_lr))
print("test accuracy score: ", accuracy_score(y_i_test,y_i_pred_test_lr))


# In[279]:


print("confusion matrix: ", "\n", confusion_matrix(y_i_test,y_i_pred_test_lr))


# In[283]:


print("classification report: ", "\n", classification_report(y_i_test,y_i_pred_test_lr))


# In[285]:


print("f1 score: ", "\n", f1_score(y_i_test,y_i_pred_test_lr))


# In[289]:


# Logistic Regression with PCA

t_lr_s     = time.time() 

gs_pca     = GridSearchCV(logit,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_pca.fit(X_pca_train,y_i_pca_train)

t_lr_e     = time.time()


print("time taken for logit with PCA: ",(t_lr_e - t_lr_s))


# In[290]:


gs_pca.best_score_


# In[291]:


gs_pca.best_params_


# In[292]:


lr_pca = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 1, multi_class = 'auto')


# In[293]:


lr_pca.fit(X_pca_train,y_i_pca_train)


# In[294]:


lr_pca.intercept_


# In[295]:


lr_pca.coef_


# In[296]:


y_i_pred_train_lr_pca = lr_pca.predict(X_pca_train)
y_i_pred_test_lr_pca  = lr_pca.predict(X_pca_test)


# In[297]:


print(y_i_pred_train_lr_pca[0:100])
print(y_i_pred_test_lr_pca[0:100])


# In[298]:


print("training accuracy score with PCA: ", accuracy_score(y_i_pca_train,y_i_pred_train_lr_pca))
print("test accuracy score with PCA: ", accuracy_score(y_i_pca_test,y_i_pred_test_lr_pca))
print("\n")

print("confusion matrix with PCA: ", "\n", confusion_matrix(y_i_pca_test,y_i_pred_test_lr_pca))

print("\n")

print("classification report with PCA: ", "\n", classification_report(y_i_pca_test,y_i_pred_test_lr_pca),"\n")

print("f1 score with PCA: ", "\n", f1_score(y_i_pca_test,y_i_pred_test_lr_pca))


# In[310]:


# KNN for all attributes

from sklearn.neighbors import KNeighborsClassifier

knn             = KNeighborsClassifier(algorithm = 'auto')

n_neighbors     = [5,10,25,50,100,200,500]

weights         = ['uniform','distance']

p               = [1,2,3] 

param_grid = [{'n_neighbors' : n_neighbors,'weights': weights,'p': p}]


# In[311]:


# Model Selection by hyperparameter tuning

t_knn_s     = time.time() 

gs_knn      = GridSearchCV(knn,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_knn.fit(X_train,y_i_train)

t_knn_e     = time.time()


print("time taken for knn with all attributes: ",(t_knn_e - t_knn_s))


# In[312]:


gs_knn.best_score_


# In[313]:


gs_knn.best_params_


# In[329]:


knn_0 = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 5, p = 1, weights = 'distance') 


# In[335]:


knn_0.fit(X_train,y_i_train)


# In[336]:


y_i_train_pred  = knn_0.predict(X_train)
y_i_test_pred = knn_0.predict(X_test) 


# In[337]:


print(y_i_train_pred[0:10])
print(y_i_test_pred[0:10])


# In[338]:


print("training accuracy score for KNN : ", accuracy_score(y_i_train,y_i_train_pred))
print("test accuracy score for KNN: ", accuracy_score(y_i_test,y_i_test_pred))
print("\n")

print("confusion matrix for KNN: ", "\n", confusion_matrix(y_i_test,y_i_test_pred))

print("\n")

print("classification report for KNN: ", "\n", classification_report(y_i_test,y_i_test_pred),"\n")

print("f1 score for KNN: ", "\n", f1_score(y_i_test,y_i_test_pred))


# In[339]:


knn_1 = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 5, p = 1, weights = 'uniform')


# In[340]:


knn_1.fit(X_train,y_i_train)


# In[341]:


knn_1.score(X_train,y_i_train)


# In[342]:


knn_1.score(X_test,y_i_test)


# In[343]:


y_i_train_pred  = knn_1.predict(X_train)
y_i_test_pred = knn_1.predict(X_test) 


# In[344]:


print("training accuracy score for KNN : ", accuracy_score(y_i_train,y_i_train_pred))
print("test accuracy score for KNN: ", accuracy_score(y_i_test,y_i_test_pred))
print("\n")

print("confusion matrix for KNN: ", "\n", confusion_matrix(y_i_test,y_i_test_pred))

print("\n")

print("classification report for KNN: ", "\n", classification_report(y_i_test,y_i_test_pred),"\n")

print("f1 score for KNN: ", "\n", f1_score(y_i_test,y_i_test_pred))


# In[345]:


# KNN with PCA

t_knn_s_pca     = time.time() 

gs_knn_pca      = GridSearchCV(knn,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_knn_pca.fit(X_pca_train,y_i_pca_train)

t_knn_e_pca     = time.time()


print("time taken for knn with all attributes: ",(t_knn_e_pca - t_knn_s_pca))


# In[346]:


gs_knn_pca.best_score_


# In[347]:


gs_knn_pca.best_params_


# In[354]:


knn_2 = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 5, p = 1, weights = 'distance') 


# In[349]:


knn_2.fit(X_pca_train,y_i_pca_train)


# In[350]:


knn_2.score(X_pca_train,y_i_pca_train)


# In[351]:


knn_2.score(X_pca_test,y_i_pca_test)


# In[353]:


knn_3 = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 5, p = 1, weights = 'uniform')


# In[355]:


knn_3.fit(X_pca_train,y_i_pca_train)


# In[357]:


y_i_train_pred_pca  = knn_3.predict(X_pca_train)
y_i_test_pred_pca   = knn_3.predict(X_pca_test) 


# In[358]:


print("training accuracy score for KNN with pca : ", accuracy_score(y_i_pca_train,y_i_train_pred_pca))
print("test accuracy score for KNN with pca: ", accuracy_score(y_i_pca_test,y_i_test_pred_pca))
print("\n")

print("confusion matrix for KNN with pca: ", "\n", confusion_matrix(y_i_pca_test,y_i_test_pred_pca))

print("\n")

print("classification report for KNN with pca: ", "\n", classification_report(y_i_pca_test,y_i_test_pred_pca),"\n")

print("f1 score for KNN with pca: ", "\n", f1_score(y_i_pca_test,y_i_test_pred_pca))


# In[359]:


# Decision Tree Classifier for all attributes

from sklearn.tree import DecisionTreeClassifier

dtc             =  DecisionTreeClassifier(random_state = 0)

criterion       = ["gini","entropy"]

max_depth       = [10,20,50,100,200,500] 

param_grid      = [{'criterion' : criterion,'max_depth': max_depth}]


# In[360]:


t_dtc_s     = time.time() 

gs_dtc      = GridSearchCV(dtc,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_dtc.fit(X_train,y_i_train)

t_dtc_e     = time.time()


print("time taken for dtc with all attributes: ",(t_dtc_e - t_dtc_s))


# In[361]:


gs_dtc.best_score_


# In[362]:


gs_dtc.best_params_


# In[549]:


dtc_0  = DecisionTreeClassifier(random_state = 0, criterion = 'gini', max_depth = 20)  


# In[550]:


dtc_0.fit(X_train,y_i_train)


# In[551]:


dtc_0.score(X_train,y_i_train)


# In[552]:


dtc_0.score(X_test,y_i_test)


# In[553]:


y_i_train_pred = dtc_0.predict(X_train)
y_i_test_pred  = dtc_0.predict(X_test)


# In[554]:


print("training accuracy score for dtc : ", accuracy_score(y_i_train,y_i_train_pred))
print("test accuracy score for dtc: ", accuracy_score(y_i_test,y_i_test_pred))
print("\n")

print("confusion matrix for dtc: ", "\n", confusion_matrix(y_i_test,y_i_test_pred))

print("\n")

print("classification report for dtc: ", "\n", classification_report(y_i_test,y_i_test_pred),"\n")

print("f1 score for dtc: ", "\n", f1_score(y_i_test,y_i_test_pred))


# In[ ]:





# In[ ]:


# DTC with PCA


# In[380]:


t_dtc_s_pca     = time.time() 

gs_dtc_pca      = GridSearchCV(dtc,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_dtc_pca.fit(X_pca_train,y_i_pca_train)

t_dtc_e_pca     = time.time()


print("time taken for dtc with pca: ",(t_dtc_e_pca - t_dtc_s_pca))


# In[381]:


gs_dtc_pca.best_score_


# In[382]:


gs_dtc_pca.best_params_


# In[555]:


dtc_1 = DecisionTreeClassifier(random_state = 0, criterion = "entropy", max_depth = 10)


# In[556]:


dtc_1.fit(X_pca_train,y_i_pca_train)


# In[557]:


dtc_1.score(X_pca_train,y_i_pca_train)


# In[558]:


dtc_1.score(X_pca_test,y_i_pca_test)


# In[560]:


y_i_train_pred = dtc_1.predict(X_pca_train)
y_i_test_pred  = dtc_1.predict(X_pca_test)


# In[561]:


print("training accuracy score for dtc with pca : ", accuracy_score(y_i_pca_train,y_i_train_pred))
print("test accuracy score for dtc with pca : ", accuracy_score(y_i_pca_test,y_i_test_pred))
print("\n")

print("confusion matrix for dtc with pca : ", "\n", confusion_matrix(y_i_pca_test,y_i_test_pred))

print("\n")

print("classification report for dtc with pca : ", "\n", classification_report(y_i_pca_test,y_i_test_pred),"\n")

print("f1 score for dtc with pca : ", "\n", f1_score(y_i_pca_test,y_i_test_pred))


# In[ ]:





# In[ ]:





# In[398]:


# SVM with all attributes

from sklearn.svm import SVC

svc        = SVC(random_state = 0)

gamma      = ['auto','scale'] 

param_grid = [{'gamma': gamma}]


# In[399]:


t_svc_s     = time.time() 

gs_svc      = GridSearchCV(svc,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_svc.fit(X_train,y_i_train)

t_svc_e     = time.time()


print("time taken for svc with all attributes: ",(t_svc_e - t_svc_s))


# In[400]:


gs_svc.best_score_


# In[401]:


gs_svc.best_params_


# In[402]:


svc_0 = SVC(random_state = 0, gamma ='auto')


# In[403]:


svc_0.fit(X_train,y_i_train)


# In[404]:


svc_0.score(X_train,y_i_train)


# In[405]:


svc_0.score(X_test,y_i_test)


# In[563]:


y_i_train_pred = svc_0.predict(X_train)
y_i_test_pred  = svc_0.predict(X_test)


# In[564]:


print("training accuracy score for svc : ", accuracy_score(y_i_train,y_i_train_pred))
print("test accuracy score for svc: ", accuracy_score(y_i_test,y_i_test_pred))
print("\n")

print("confusion matrix for svc: ", "\n", confusion_matrix(y_i_test,y_i_test_pred))

print("\n")

print("classification report for svc: ", "\n", classification_report(y_i_test,y_i_test_pred),"\n")

print("f1 score for svc: ", "\n", f1_score(y_i_test,y_i_test_pred))


# In[ ]:





# In[406]:


# Random Forest Classifier - ensembling, with all attributes

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)

n_estimators = [10,25,50,100,200,300,500]

criterion    = ['gini','entropy'] 

max_depth    = [10,20,50,100,200,500]

param_grid   = [{'n_estimators': n_estimators, 'criterion': criterion, 
                 'max_depth': max_depth}]


# In[407]:


t_rf_s     = time.time() 

gs_rf      = GridSearchCV(RF,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_rf.fit(X_train,y_i_train)

t_rf_e     = time.time()


print("time taken for RF with all attributes: ",(t_rf_e - t_rf_s))


# In[408]:


gs_rf.best_score_


# In[409]:


gs_rf.best_params_


# In[505]:


rf_0  =  RandomForestClassifier(random_state = 0, criterion = 'entropy', n_estimators = 100, max_depth = 20)


# In[506]:


rf_0.fit(X_train,y_i_train)


# In[507]:


rf_0.score(X_train,y_i_train)


# In[508]:


rf_0.score(X_test,y_i_test)


# In[510]:


y_i_train_pred  = rf_0.predict(X_train)
y_i_test_pred   = rf_0.predict(X_test)


# In[513]:


print("training accuracy score for RF : ", accuracy_score(y_i_train,y_i_train_pred))
print("test accuracy score for RF : ", accuracy_score(y_i_test,y_i_test_pred))
print("\n")

print("confusion matrix for RF : ", "\n", confusion_matrix(y_i_test,y_i_test_pred))

print("\n")

print("classification report for RF : ", "\n", classification_report(y_i_test,y_i_test_pred),"\n")

print("f1 score for RF : ", "\n", f1_score(y_i_test,y_i_test_pred))


# In[438]:


feat_labels = cs_df.columns[:-1]

importances = rf_0.feature_importances_

indices     = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print ("%2d %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))


# In[514]:


plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align ='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])


# In[440]:


rf_1  =  RandomForestClassifier(random_state = 0, criterion = 'entropy', n_estimators = 100, max_depth = 20)


# In[441]:


rf_1.fit(X_pca_train,y_i_pca_train)


# In[442]:


rf_1.score(X_pca_train,y_i_pca_train)


# In[443]:


rf_1.score(X_pca_test,y_i_pca_test)


# In[ ]:


# feature selection to decrease overfitting

# Feature selection using correlation heatmap

# Variables that can be removed because of correlation for feature selection

# from scatter plots 1 -  




# ('EBITDA', 'EBITDA Margin'), 
# ('Net Income Before Extras', 'ROA', 'ROE'), 
# ('Current Liquidity', 'Cash')




# these combinations are correlated, so out of these 7 variables, 3 variables can be selected :
# selection based on correlation with 'InvGrd' & 'Rating' : 
# selected ones for 'InvGrd' : ('EBITDA','Net Income Before Extras', 'Current Liquidity')
# selected ones for 'Rating' : ('EBITDA', 'ROE', 'Current Liquidity')

# from scatter plots 2 - 



# ('Total Debt','LT Debt','Total Debt/EBITDA','Total Debt/MV')
# ('Net Debt/EBITDA','Net Debt/MV', 'Net Debt')
# ('CFO','CFO/Debt')





# these combinations are correlated, so out of these 7 variables, 2 variables can be selected :
# selected ones for 'InvGrd' : ('Total Debt','Net Debt')
# selected ones for 'Rating' : ('Total Debt/MV','Net Debt/MV')

# So from these 14 variables, we can select 5 variables for our model, without loss of explained variance
# (we have to check using evaluation metrics, whether this feature selection gives best results


# In[444]:


cs_df.columns


# In[477]:


# based on correlation heatmap and Random forest feature importance

features = ['Free Cash Flow','ROA','CFO','EPS Before Extras','EBITDA','Gross Margin','PE']
            
    
#,'Current Liquidity','Total MV','Sales/Revenues','Total Debt/MV','Net Debt/MV','Current Liabilities',
#'Total Liquidity','ST Debt','Interest Coverage']


# In[478]:


len(features)


# In[479]:


X_f = X[features]


# In[480]:


X_f.shape


# In[481]:


X_f.head()


# In[482]:


X_f.tail()


# In[483]:


X_f_train,X_f_test,y_i_f_train,y_i_f_test = train_test_split(X_f,y_i,test_size = 0.25,random_state = 42, 
                                                             stratify = y_i)


# In[484]:


rf_2  =  RandomForestClassifier(random_state = 0, criterion = 'entropy', n_estimators = 100, max_depth = 20)


# In[485]:


rf_2.fit(X_f_train,y_i_f_train)


# In[486]:


rf_2.score(X_f_train,y_i_f_train)


# In[487]:


rf_2.score(X_f_test,y_i_f_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[517]:


cs_df.head()


# In[516]:


cs_df_rp.head()


# In[518]:


X_1_train,X_1_test,y_r_train,y_r_test = train_test_split(X,y_r,test_size = 0.1,random_state = 42, 
                                                             stratify = y_r)


# In[521]:


y_r_train[0:5]


# In[522]:


# Multi class classification

# Decision Tree Classifier


dtc             =  DecisionTreeClassifier(random_state = 0)

criterion       = ["gini","entropy"]

max_depth       = [10,20,50,100,200,500] 

param_grid      = [{'criterion' : criterion,'max_depth': max_depth}]


# In[524]:


# Model Selection by hyperparameter tuning

t_dtc_s     = time.time() 

gs_dtc      = GridSearchCV(dtc,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_dtc.fit(X_1_train,y_r_train)

t_dtc_e     = time.time()


print("time taken for dtc with all attributes: ",(t_dtc_e - t_dtc_s))


# In[525]:


gs_dtc.best_score_


# In[526]:


gs_dtc.best_params_


# In[542]:


dtc_0 = DecisionTreeClassifier(random_state = 0, criterion = 'entropy', max_depth = 20)


# In[567]:


dtc_0.fit(X_1_train,y_r_train)


# In[568]:


dtc_0.score(X_1_train,y_r_train)


# In[569]:


dtc_0.score(X_1_test,y_r_test)


# In[ ]:





# In[ ]:





# In[574]:


logit      = LogisticRegression(multi_class = 'auto')

penalty    = ['l1','l2']

c          = [100,10,1,0.1,0.01,0.001]

param_grid = [{'penalty' : penalty,'C': c}]


# In[575]:


# Model Selection by hyperparameter tuning

t_lr_s     = time.time() 

gs = GridSearchCV(logit,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs.fit(X_1_train,y_r_train)

t_lr_e     = time.time()


print("time taken for logit with all attributes: ",(t_lr_e - t_lr_s))


# In[576]:


gs.best_score_


# In[577]:


gs.best_params_


# In[583]:


lr_mc = LogisticRegression(multi_class = 'auto', C = 100, penalty = 'l1')


# In[584]:


lr_mc.fit(X_1_train,y_r_train)


# In[585]:


lr_mc.score(X_1_train,y_r_train)


# In[586]:


lr_mc.score(X_1_test,y_r_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[533]:


# KNN for all attributes

from sklearn.neighbors import KNeighborsClassifier

knn             = KNeighborsClassifier(algorithm = 'auto')

n_neighbors     = [5,10,25,50,100,200,500]

weights         = ['uniform','distance']

p               = [1,2,3] 

param_grid = [{'n_neighbors' : n_neighbors,'weights': weights,'p': p}]


# In[534]:


t_knn_s     = time.time() 

gs_knn      = GridSearchCV(knn,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_knn.fit(X_1_train,y_r_train)

t_knn_e     = time.time()


print("time taken for knn with all attributes: ",(t_knn_e - t_knn_s))


# In[535]:


gs_knn.best_score_


# In[536]:


gs_knn.best_params_


# In[578]:


knn_mc = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 5, p = 1, weights = 'distance')


# In[579]:


knn_mc.fit(X_1_train,y_r_train)


# In[580]:


knn_mc.score(X_1_train,y_r_train)


# In[581]:


knn_mc.score(X_1_test,y_r_test)


# In[ ]:





# In[593]:


# SVM with all attributes

from sklearn.svm import LinearSVC

svc        = LinearSVC(random_state = 0,multi_class = 'ovr')


# In[594]:


svc.fit(X_1_train,y_r_train)


# In[595]:


svc.score(X_1_train,y_r_train)


# In[596]:


svc.score(X_1_test,y_r_test)


# In[ ]:





# In[ ]:





# In[537]:


# Random Forest Classifier - ensembling, with all attributes

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)

n_estimators = [10,25,50,100,200,300,500]

criterion    = ['gini','entropy'] 

max_depth    = [10,20,50,100,200,500]

param_grid   = [{'n_estimators': n_estimators, 'criterion': criterion, 
                 'max_depth': max_depth}]


# In[538]:


t_rf_s     = time.time() 

gs_rf      = GridSearchCV(RF,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs_rf.fit(X_1_train,y_r_train)

t_rf_e     = time.time()


print("time taken for RF with all attributes: ",(t_rf_e - t_rf_s))


# In[539]:


gs_rf.best_score_


# In[540]:


gs_rf.best_params_


# In[597]:


rf_mc = RandomForestClassifier(random_state = 0, n_estimators = 300, criterion = 'gini', max_depth = 50)


# In[598]:


rf_mc.fit(X_1_train,y_r_train)


# In[599]:


rf_mc.score(X_1_train,y_r_train)


# In[600]:


rf_mc.score(X_1_test,y_r_test)


# In[601]:


# END


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




