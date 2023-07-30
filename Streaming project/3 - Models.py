#!/usr/bin/env python
# coding: utf-8

# # Models

# DATASET INFORMATION:
# 
# Predictive variables:
# 
# - 'league': volleyball match league
# - "league_level": league classification (1: higher classification, 4: lower classification).
# - "country": country were the volleyball match is played
# - "continent": continent were the volleyball match is played
# - 'num_operators_day': total number of operators per date
# - 'date': match date
# - 'start_time' : time of match start
# - 'year': the year in which the match took place
# - 'months': the month in which the match took place
# - 'rank_week': week number (variables are 1 (first week), 2(second week), 3(third week), and 4 (fourth week)).
# - 'day': day of the month in which the match took place
# - 'weekday': day of the week in which the match took place
# - 'day_time': Grouped time ranges (early morning, morning, afternoon, night)
# - freq_rank_week_day_time: matches grouped per weekday and time
# - freq_month_day_time: matches grouped per month and time
# - 2h_interval: Match time duration
# - GLnum_matches_interval: Volleyball matches grouped per date and time interval
# - Totalnum_matches_interval: total number of matches streamed by different streaming productions from different disciplines per date and time
# 
# Target variable:
# 
# - 'issue': streaming problems (1= streaming problems, 0= no streaming problems)

# In[1]:


import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#from dataprep.eda import create_report
from sklearn.metrics import accuracy_score,plot_confusion_matrix,roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import precision_score
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif,RFECV,RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,plot_confusion_matrix,roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc


# In[2]:


#Dataset imported
data = pd.read_csv('data_4_models.csv',encoding='unicode_escape')
data.head(5)


# In[3]:


#Deletion of columns that will not be used for the analysis
data = data.drop(['Unnamed: 0',"day","start_time", "date",'day_time','continent'], axis=1)

data.head()


# In[4]:


# Final check of null values and type of variables
data.info()


# In[5]:


# list of categorical variables 

categorical_columns = [col for col in data.columns if data[col].dtypes == 'object']

categorical_columns


# In[6]:


# List of quantitative variables selected

numerical_columns = [col for col in data.columns if data[col].dtypes != 'object']

numerical_columns


# ### Train Test split

# In[7]:


# train test split
X = data.drop(['issue'], axis=1)
y = data['issue']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 1986)


# In[8]:


print('Size of training set:')
print(X_train.shape)

print('Y train values:')
print(y_train.value_counts(normalize=False))


print('Relative frequencies of Y train values')
print(y_train.value_counts(normalize=True))


# ### Dummy variables

# In[9]:


#List of values of predictive categorical variables

encoder_categories = []

X_categorical_columns = [x for x in categorical_columns]

for col in categorical_columns:    
    col_categories = data[col].unique()
    encoder_categories.append(col_categories)

encoder_categories


# In[10]:


# Categorical predictive variables transformed into dummy variables
encoder = OneHotEncoder(categories = encoder_categories,drop='first', sparse=False)

encoder = encoder.fit(X_train[X_categorical_columns])

   
X_train_encoded = encoder.transform(X_train[X_categorical_columns])
X_train_categorical = pd.DataFrame(X_train_encoded, columns = encoder.get_feature_names(X_categorical_columns))


X_test_encoded = encoder.transform(X_test[X_categorical_columns])
X_test_categorical = pd.DataFrame(X_test_encoded, columns = encoder.get_feature_names(X_categorical_columns))
X_test_categorical.head()


# ### Standarization of numerical variables

# In[11]:


# list of numerical predictive variables 
X_numerical_columns = [x for x in numerical_columns if x != 'issue']
X_numerical_columns


# In[12]:


# Standardization of numerical predictive variables
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[X_numerical_columns])
X_train_numerical = pd.DataFrame(X_train_scaled, columns = X_numerical_columns)

X_test_scaled = scaler.transform(X_test[X_numerical_columns])
X_test_numerical = pd.DataFrame(X_test_scaled, columns = X_numerical_columns)
X_test_numerical


# In[13]:


# index reset
#standardized variables and dummy variables grouped in train and test datasets

Y_train_concat = y_train.reset_index(drop=True)
Y_test_concat = y_test.reset_index(drop=True)

data_train = pd.concat([X_train_categorical, X_train_numerical, Y_train_concat], axis=1)

data_test = pd.concat([X_test_categorical, X_test_numerical, Y_test_concat], axis=1)

data_total= pd.concat([data_train,data_test],axis=0)

x_data_total = data_total.drop(['issue'],axis=1)
y_data_total = data_total['issue']

# final train and test datasets for predictive variables (X) and target variable (y)
x_train_last = data_train.drop(['issue'],axis=1)
y_train_last = data_train['issue']

x_test_last = data_test.drop(['issue'],axis=1)
y_test_last = data_test['issue']


# In[14]:


# Size of train and test datasets
print(x_train_last.shape)
print(y_train_last.shape)
print(x_test_last.shape)
print(y_test_last.shape)
print(data_total.shape)


# ### Balancing the dataset with oversampling (SMOTENC)

# In[15]:


categorical_cols=(X_train.dtypes=='object').values
numerical_cols= ~categorical_cols

sm=SMOTENC(categorical_features=categorical_cols ,sampling_strategy='minority')
X_train_smo,y_train_smo=sm.fit_resample(x_train_last,y_train_last)

print('X_train_smo:',X_train_smo.shape)
print('Y Train balanced classes:')
print(y_train_smo.value_counts())

print('\n\nX_test:',x_test_last.shape)
print('Y test classes:')
print(y_test_last.value_counts())


# In[16]:


#Unbalanced classes of "issue" variable
count_classes = data.value_counts(y_train, sort = True)
count_classes.plot(kind = 'bar', rot=0, color= "pink")
plt.xticks(range(2))
plt.title("Frequencies of issue variable")
plt.xlabel("issue")
plt.ylabel("Number of observations");


# In[17]:


#Balanced classes of "issue" variable
count_classes = y_train_smo.value_counts( sort = True)
count_classes.plot(kind = 'bar', rot=0, color= "pink")
plt.xticks(range(2))
plt.title("Frequencies of issue variable")
plt.xlabel("issue")
plt.ylabel("Number of observations");


# ### Feature selection - Reduction of dimensionality

# ### SelectKBest

# In[18]:


# Features selection according to the k highest scores with balanced dataset 

skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=1986)
steps=([('selector',SelectKBest(f_classif)),('classif',RandomForestClassifier(class_weight='balanced_subsample'))])
pipe=Pipeline(steps)

# gridsearchCV to select parameter k
param_grid={'selector__k':np.arange(10,150,20)}
grid=GridSearchCV(pipe,param_grid,scoring='recall',cv=skf,verbose=3,n_jobs=3)
grid.fit(X_train_smo,y_train_smo) 


# In[19]:


# Results
n_features=grid.cv_results_['param_selector__k']
mn_cv_score=grid.cv_results_['mean_test_score']
err=grid.cv_results_['std_test_score']
plt.bar(n_features,mn_cv_score,color = "r",width=3,yerr=err,align = "center")
plt.xlabel('Number of features')
plt.ylabel('Test score');


# In[20]:


mn_cv_score


# In[21]:


# Function calculates metrics to evaluate a classificator 

def evaluate_model(model,X,y_true):
    
    y_pred=model.predict(X)
    y_proba=model.predict_proba(X)

    print(classification_report(y_true,y_pred))
    print('Area under ROC curve:',np.round(roc_auc_score(y_true,y_proba[:,1]),4))
    precision, recall,threshold=precision_recall_curve(y_true,y_proba[:,1]);
    print('Area under Precision-Recall curve:',np.round(auc(recall,precision),4))
    plot_confusion_matrix(model,X,y_true,cmap='Blues');
    return


# In[22]:


# Evaluation with test data 
skb=SelectKBest(f_classif,k=30)
X_train_reduced=skb.fit_transform(X_train_smo,y_train_smo)
X_test_reduced=x_test_last.loc[:,skb.get_support()]
model=RandomForestClassifier(class_weight='balanced_subsample')
model.fit(X_train_reduced,y_train_smo)
evaluate_model(model,X_test_reduced,y_test_last)


# In[23]:


# Feature importance (computed as the mean and standard deviation of accumulation of the impurity decrease within each tree)
y=np.sort(model.feature_importances_)
x=np.argsort(model.feature_importances_)
x=x[::-1]
feat_names =X_train_smo.columns[skb.get_support()]
labels=feat_names[x]
y=y[::-1]

plt.figure(figsize=(15,8))
plt.bar(range(len(y)),y,color = "r",width=3,align = "center")
plt.xticks(range(len(y)), labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Mean decrease in impurity');

plt.xlim([0,30])


# In[24]:


#Final dataset with 30 nfeatures selected by SelectKBest method
X_train_kbest = pd.DataFrame(X_train_smo, columns = feat_names)
X_test_kbest = pd.DataFrame(x_test_last, columns = feat_names)

print("X train", X_train_kbest.shape)
print("Y train", y_train_smo.shape)
print("X test", X_test_kbest.shape)
print("Y test", y_test_last.shape)


# # Models - Part 1

# In[25]:


# models
dt = DecisionTreeClassifier(random_state=1)

models = [LogisticRegression(),
          KNeighborsClassifier(),
          DecisionTreeClassifier(),
          AdaBoostClassifier(base_estimator=dt,random_state=1986),
          GradientBoostingClassifier()
]


# In[26]:


# parameters
params = [
    {'C': [1, 10, 100, 1000],
     'penalty': ['l1', 'l2',],
     'solver': ['saga'],
    "max_iter": [10000]},
    {'n_neighbors': range(1,10),
     'weights' : ['uniform', 'distance'],
     'p' : [1, 2, 3]},
    {"criterion" : ["gini", "entropy"],
                "min_samples_leaf": [5,10,None], 
                "max_depth" : [1,2,3,4,5,6,8,9,10,None],
                "min_samples_split": [2, 3,None]},
    {"n_estimators": [500,1000],
          "learning_rate":[0.01, 0.1],
        "base_estimator__max_depth": [1, 2, 3]},
    {'n_estimators':[500, 1000] , 
             'learning_rate':[0.001, 0.001, 0.1],
            'max_depth' : [1, 2, 3, 4]}
]


# In[27]:


#cross validation with StratifiedKFold
# Exhaustive search over parameters values with GridSearchCV
folds=StratifiedKFold(n_splits=5, random_state=1986, shuffle=True)
grids = []
for i in range(len(models)):
    gs = GridSearchCV(estimator=models[i], param_grid=params[i], scoring='recall', cv=folds, n_jobs=3, verbose=1)
    print (gs)
    fit = gs.fit(X_train_kbest,y_train_smo)
    grids.append(fit)


# In[28]:


# Results:
# best score: Mean cross-validated score of the best_estimator
# best estimator: Estimator that was chosen by the search (estimator which gave highest score or smallest loss if specified, on the left out data).
# best params: Parameter setting that gave the best results on the hold out data.
for i in grids:
    print (i.best_score_)
    print (i.best_estimator_)
    print (i.best_params_)


# In[29]:


# Model´s predicted labels y:
# y_preds_log : predictions of Logistic Regression classifier
# y_preds_knn: predictions of K Neighbors classifier
# y_preds_tree: predictions of Decision Tree Classifier
# y_preds_ABc: predictions of AdaBoost Classifier
# y_preds_GBc: predictions of Gradient Boosting Classifier

y_preds_log = grids[0].predict(X_test_kbest)
y_preds_knn = grids[1].predict(X_test_kbest)
y_preds_tree = grids[2].predict(X_test_kbest)
y_preds_ABc = grids[3].predict(X_test_kbest)
y_preds_GBc = grids[4].predict(X_test_kbest)


# ### Ensemble model

# In[30]:


# Gaussian Naive Bayes model
gnb = GaussianNB()

model_gnb= gnb.fit(X_train_kbest,y_train_smo)  

Y_pred_gnb = gnb.predict(X_test_kbest)


# In[31]:


# Best estimators obtained with previous Grid Search Cross Validation
# bestestimator_rl --> Logistic Regression classifier
# bestestimator_knn --> K Neighbors classifier
# bestestimator_tree --> Decision Tree Classifier
bestestimator_rl = grids[0].best_estimator_
bestestimator_knn = grids[1].best_estimator_
bestestimator_tree = grids[2].best_estimator_
print(bestestimator_rl)
print(bestestimator_knn)
print(bestestimator_tree)


# In[32]:


# Best parameters obtained with previous Grid Search Cross Validation
# bestesparams_rl --> Logistic Regression classifier
# bestparams_knn --> K Neighbors classifier
# bestparams_tree --> Decision Tree Classifier
bestparams_rl = grids[0].best_params_
bestparams_knn = grids[1].best_params_
bestparams_tree = grids[2].best_params_
print(bestparams_rl)
print(bestparams_knn)
print(bestparams_tree)


# In[33]:


#final models (Logistic Regression classifier, K Neighbors classifier, Decision Tree Classifier)
model_rl = LogisticRegression(C=100, penalty= "l2", solver='saga')
model_knn = KNeighborsClassifier(n_neighbors= 5, p= 3, weights= "uniform")
model_tree = DecisionTreeClassifier(criterion='entropy', max_depth= None, min_samples_leaf=5,
                       min_samples_split=3)


# In[34]:


model_rl.fit(X_train_kbest,y_train_smo)
model_knn.fit(X_train_kbest,y_train_smo)
model_tree.fit(X_train_kbest,y_train_smo)


# In[35]:


#ensemble model combining models:
# Gaussian Naive Bayes, Logistic Regression classifier, K Neighbors classifier, Decision Tree Classifier
def predict_ensemble(X, gnb, model_lr, model_knn, model_tree):
    y_pred_1 = gnb.predict_proba(X)[:, 1]
    y_pred_2 = model_rl.predict_proba(X)[:, 1]
    y_pred_3 = model_knn.predict_proba(X)[:, 1]
    y_pred_4 = model_tree.predict_proba(X)[:, 1]
    result = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4) / 4
    return result


# In[36]:


# prediction results for Ensemble and Gaussian Naive Bayes models
y_preds_ensemble = predict_ensemble(X_test_kbest, gnb, model_rl, model_knn, model_tree)
threshold = 0.5
y_preds_ensemble = [1 if (x >= threshold) else 0 for x in y_preds_ensemble] 
Y_preds_gnb = gnb.predict(X_test_kbest)


# ### Model evaluation metrics 

# **Accuracy**
# 
# (Number of correct predictions/total number of samples)

# In[37]:


accuracy_knn = accuracy_score(y_test, y_preds_knn)
accuracy_bayes = accuracy_score(y_test_last, Y_preds_gnb)
accuracy_tree = accuracy_score(y_test_last, y_preds_tree)
accuracy_lr = accuracy_score(y_test_last, y_preds_log)
accuracy_ensemble = accuracy_score(y_test_last, y_preds_ensemble)
acurracy_ABc = accuracy_score(y_test_last, y_preds_ABc)
acurracy_GBc = accuracy_score(y_test_last,y_preds_GBc)
print('Accuracy KNN =',accuracy_knn)
print('Accuracy Bayes =',accuracy_bayes)
print('Accuracy tree =', accuracy_tree)
print("Accuracy lr =",accuracy_lr)
print("Accuracy ensamble=", accuracy_ensemble)
print("Accuracy ABc=", acurracy_ABc)
print("Acurracy_GBc", acurracy_GBc)


# **Recall**
# 
# The recall is intuitively the ability of the classifier to find all the positive samples:
# True positives / (true positives + false negatives)

# In[38]:


from sklearn.metrics import recall_score
recall_knn = recall_score(y_test_last, y_preds_knn)
recall_bayes = recall_score(y_test_last, Y_preds_gnb)
recall_tree = recall_score(y_test_last, y_preds_tree)
recall_lr = recall_score(y_test_last, y_preds_log)
recall_ensemble = recall_score(y_test_last, y_preds_ensemble)
recall_ABc = recall_score(y_test_last, y_preds_ABc)
recall_GBc = recall_score(y_test_last,y_preds_GBc)
print('Recall KNN =',recall_knn)
print('Recall Bayes =',recall_bayes)
print('Recall tree =', recall_tree)
print("Recall lr =",recall_lr)
print("Recall ensamble=", recall_ensemble)
print("Recall ABc=",recall_ABc)
print("Recall GBc=",recall_GBc)


# **Precision**
# 
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative:
# True positive/(true positive + false positive)

# In[39]:


precision_knn = precision_score(y_test_last, y_preds_knn)
precision_bayes = precision_score(y_test_last, Y_preds_gnb)
precision_tree = precision_score(y_test_last, y_preds_tree)
precision_lr = precision_score(y_test_last, y_preds_log)
precision_ensemble = precision_score(y_test_last, y_preds_ensemble)
precision_ABc = precision_score(y_test_last, y_preds_ABc)
precision_GBc = precision_score(y_test_last,y_preds_GBc)
print('precision KNN =',precision_knn)
print('precision Bayes =',precision_bayes)
print('precision tree =', precision_tree)
print("precision lr =",precision_lr)
print("precision ensamble=", precision_ensemble)
print("precision ABc =", precision_ABc)
print("precision GBc =", precision_GBc)


# **F1 score**
# 
# The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
# F1 = 2 * (precision * recall) / (precision + recall)

# In[40]:


f1_knn = f1_score(y_test_last, y_preds_knn)
f1_bayes = f1_score(y_test_last, Y_preds_gnb)
f1_tree = f1_score(y_test_last, y_preds_tree)
f1_lr = f1_score(y_test_last, y_preds_log)
f1_ABc = f1_score(y_test_last, y_preds_ABc)
f1_GBc = f1_score(y_test_last,y_preds_GBc)
f1_ensemble = f1_score(y_test_last, y_preds_ensemble)
print('f1 KNN =',f1_knn)
print('f1 Bayes =',f1_bayes)
print('f1 tree =', f1_tree)
print("f1 lr =",f1_lr)
print("f1 ensamble=", f1_ensemble)
print("f1 ABc =", f1_ABc)
print("f1 GBc =", f1_GBc)


# **AUC ROC score**

# In[41]:


roc_auc_knn = roc_auc_score(y_test_last, y_preds_knn)
roc_auc_bayes = roc_auc_score(y_test_last, Y_preds_gnb)
roc_auc_tree = roc_auc_score(y_test_last, y_preds_tree)
roc_auc_lr = roc_auc_score(y_test_last, y_preds_log)
roc_auc_ABc = roc_auc_score(y_test_last, y_preds_ABc)
roc_auc_GBc = roc_auc_score(y_test_last,y_preds_GBc)
roc_auc_ensemble = roc_auc_score(y_test_last, y_preds_ensemble)
print('roc_auc KNN =',roc_auc_knn)
print('roc_auc Bayes =',roc_auc_bayes)
print('roc_auc tree =', roc_auc_tree)
print("roc_auc lr =",roc_auc_lr)
print("roc_auc ensamble=", roc_auc_ensemble)
print("roc_auc ABc =", roc_auc_ABc)
print("roc_auc GBc =", roc_auc_GBc)


# ### Confusion matrix report

# In[42]:


# Prediction of class probabilies by each model
# y_preds_log_proba : Prediction of class probabilies of Logistic Regression classifier
# y_preds_knn_proba: Prediction of class probabilies of K Neighbors classifier
# y_preds_tree_proba: Prediction of class probabilies of Decision Tree Classifier
# y_preds_ABc_proba: Prediction of class probabilies of AdaBoost Classifier
# y_preds_GBc_proba: Prediction of class probabilies of Gradient Boosting Classifier
# y_preds_bayes_proba: Prediction of class probabilies of Gaussian Naive Bayes model
#y_preds_ensemble_proba: Prediction of class probabilies of Ensemble model

def predict_proba_ensemble(X, gnb, model_lr, model_knn, model_tree):
    y_pred_1 = gnb.predict_proba(X)
    y_pred_2 = model_rl.predict_proba(X)
    y_pred_3 = model_knn.predict_proba(X)
    y_pred_4 = model_tree.predict_proba(X)
    result = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4) / 4
    return result

y_preds_log_proba = grids[0].predict_proba(X_test_kbest)
y_preds_knn_proba = grids[1].predict_proba(X_test_kbest)
y_preds_tree_proba = grids[2].predict_proba(X_test_kbest)
y_preds_ABc_proba = grids[3].predict_proba(X_test_kbest)
y_preds_GBc_proba = grids[4].predict_proba(X_test_kbest)
y_preds_bayes_proba = gnb.predict_proba(X_test_kbest)
y_preds_ensemble_proba = predict_proba_ensemble(X_test_kbest, gnb, model_rl, model_knn, model_tree)


# In[43]:


# Function calculates metrics to evaluate a classificator 
def evaluate_model_last(y_test,y_pred,y_proba,name):

    conf_mat_gnb = confusion_matrix(y_test, y_pred)
    print(name+ '\n\n')
    print('Confusion matrix\n')
    confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,fmt='4d')
    plt.ylabel('Actual values')
    plt.xlabel('Predictes values');
    
    print(classification_report(y_test,y_pred))
    print('Area under ROC curve:',np.round(roc_auc_score(y_test,y_proba[:,1]),4))
    precision, recall,threshold=precision_recall_curve(y_test,y_proba[:,1]);
    print('Area under Precision-Recall curve:',np.round(auc(recall,precision),4))
    
    return


# In[44]:


evaluate_model_last(y_test_last,y_preds_knn,y_preds_knn_proba, 'Confusion Matrix: K Neighbors classifier')


# In[45]:


evaluate_model_last(y_test_last,Y_preds_gnb,y_preds_bayes_proba, 'Confusion Matrix:  Gaussian Naive Bayes model')


# In[46]:


evaluate_model_last(y_test_last,y_preds_tree,y_preds_tree_proba, 'Confusionn Matrix: Decision Tree Classifier')


# In[47]:


evaluate_model_last(y_test_last,y_preds_log,y_preds_log_proba, 'Confusion Matrix: Logistic Regression classifier')


# In[48]:


evaluate_model_last(y_test_last,y_preds_ensemble,y_preds_ensemble_proba, 'Confusion Matrix: Ensemble model')


# In[49]:


evaluate_model_last(y_test_last,y_preds_ABc,y_preds_ABc_proba, 'Confusion Matrix: AdaBoost Classifier')


# In[50]:


evaluate_model_last(y_test_last,y_preds_GBc,y_preds_GBc_proba, 'Confusion matrix: Gradient Boosting Classifier')


# **CURVA ROC**

# In[51]:


# las variables que se quedaron comentarios me están dando error 
fpr_log_log,tpr_log_log,thr_log_log = roc_curve(y_test_last, y_preds_log_proba[:,1])
fpr_log_knn,tpr_log_knn,thr_log_knn = roc_curve(y_test_last, y_preds_knn_proba[:,1])
fpr_log_bayes,tpr_log_bayes,thr_log_bayes = roc_curve(y_test_last, y_preds_bayes_proba[:,1])
fpr_log_tree,tpr_log_tree,thr_log_tree = roc_curve(y_test_last, y_preds_tree_proba[:,1]) 
fpr_log_ens,tpr_log_ens,thr_log_ens = roc_curve(y_test_last, y_preds_ensemble_proba[:,1])
fpr_log_ABc,tpr_log_ABc,thr_log_ABc = roc_curve(y_test_last, y_preds_ABc_proba[:,1])
fpr_log_GBc,tpr_log_GBc,thr_log_GBc = roc_curve(y_test_last, y_preds_GBc_proba[:,1])

auc_log = auc(fpr_log_log, tpr_log_log)
auc_knn = auc(fpr_log_knn, tpr_log_knn)
auc_bayes = auc(fpr_log_bayes, tpr_log_bayes)
auc_tree = auc(fpr_log_tree,tpr_log_tree)
auc_ens = auc(fpr_log_ens,tpr_log_ens)
auc_ABc = auc(fpr_log_ABc,tpr_log_ABc)
auc_GBc = auc(fpr_log_GBc,tpr_log_GBc)

print("AUC K Neighbors classifier", auc_knn)
print("AUC Gaussian Naive Bayes model", auc_bayes)
print("AUC Decision Tree Classifier", auc_tree)
print("AUC Logistic Regression classifier", auc_log)
print("AUC Ensemble model", auc_ens)
print("AUC AdaBoost Classifier", auc_ABc)
print("AUC Gradient Boosting Classifier", auc_GBc)


# In[52]:


df_log = pd.DataFrame(dict(fpr=fpr_log_log, tpr=tpr_log_log, thr = thr_log_log))
plt.plot(df_log['fpr'],df_log['tpr'], label='Logistic Regression classifier (area = %0.2f)' % auc_log)

df_knn = pd.DataFrame(dict(fpr=fpr_log_knn, tpr=tpr_log_knn, thr = thr_log_knn))
plt.plot(df_knn['fpr'],df_knn['tpr'], label='K Neighbors classifier (area = %0.2f)' % auc_knn)

df_bayes = pd.DataFrame(dict(fpr=fpr_log_bayes, tpr=tpr_log_bayes, thr = thr_log_bayes))
plt.plot(df_bayes['fpr'],df_bayes['tpr'], label='Gaussian Naive Bayes model (area = %0.2f)' % auc_bayes)


df_tree = pd.DataFrame(dict(fpr=fpr_log_tree, tpr=tpr_log_tree, thr = thr_log_tree))
plt.plot(df_tree['fpr'],df_tree['tpr'], label='Decision Tree Classifier (area = %0.2f)' % auc_tree)

df_ensamble = pd.DataFrame(dict(fpr=fpr_log_ens, tpr=tpr_log_ens, thr = thr_log_ens))
plt.plot(df_ensamble['fpr'],df_ensamble['tpr'], label='Ensemble model(area = %0.2f)' % auc_ens)

df_ABc = pd.DataFrame(dict(fpr=fpr_log_ABc, tpr=tpr_log_ABc, thr = thr_log_ABc))
plt.plot(df_ABc['fpr'],df_ABc['tpr'], label='AdaBoost Classifier (area = %0.2f)' % auc_ABc)

df_GBc = pd.DataFrame(dict(fpr=fpr_log_GBc, tpr=tpr_log_GBc, thr = thr_log_GBc))
plt.plot(df_GBc['fpr'],df_GBc['tpr'], label='Gradient Boosting Classifier (area = %0.2f)' % auc_GBc)


plt.axis([0, 1.01, 0, 1.01]); plt.legend()
plt.xlabel('1 - Specificty'); plt.ylabel('TPR / Sensitivity'); plt.title('ROC Curve')
plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))


plt.show()


# ### As per model metrics, K Neighbors classifier and AdaBoost classifier have good performances.

# ## Models - Part 2 - Time series

# In[53]:


#Dataset imported
data = pd.read_csv('data_4_models.csv',encoding='unicode_escape')
data.head(5)


# In[54]:


data.columns


# In[55]:


#Deletion of columns that will not be used for the analysis
#creation of dataset data_1:
data_1 = data.drop(['Unnamed: 0','continent','league_level','num_operators_day','rank_week','GLnum_matches_interval','Totlnum_match_y_m_wd_int','day','start_time','year','league','day_time','freq_weekday_day_time','freq_rank_week_day_time','freq_month_day_time','start_time_interval','country'], axis=1)
data_1


# In[56]:


#convertion of variable "date" from string into DateTime.
data_1['date'] = pd.to_datetime(data_1['date'], format='%Y/%m/%d')
data_1.info()


# In[57]:


#Creation of new variables: "Year", "Month" 
data_1['Year'] = data_1['date'].dt.year 
data_1['Month'] = data_1['date'].dt.month 
data_1


# In[58]:


#Creation of new variable "month_year"
data_1['year_str'] = data_1.Year.astype(str)
data_1['month_str'] = data_1.Month.astype(str)
data_1['month_year'] = data_1.month_str.str.cat(data_1.year_str, sep='/')

data_1['month_year'] = pd.to_datetime(data_1['month_year'], format='%m/%Y')

#new dataset "data_2"
data_2= data_1.drop(['date','Year','Month','year_str','month_str'],axis=1)
data_2


# In[59]:


#average issues grouped per year and month:
frequency_issue = data_2.groupby(['month_year'])
group_frequency_issue= pd.DataFrame(frequency_issue["issue"].mean())
print(group_frequency_issue)
data_merge=pd.merge(data_2,group_frequency_issue, on=['month_year'])
data_merge.rename(columns={'issue_y':'issue_mean'}, inplace=True)
data_merge_2= data_merge.drop_duplicates(subset=['month_year'])

data_merge_2


# In[60]:


#Index reset per year and month.
data_merge_3 = data_merge_2.sort_values(by='month_year')
data_merge_4 = data_merge_3.reset_index()
data_final = data_merge_4.drop(['index'],axis=1)
data_final


# In[61]:


# Funtion to plot time series:
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Issue mean', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[62]:


#Graph that shows issue mean through time
plot_df(data_final, x=data_final.month_year, y=data_final.issue_mean, title="Issue mean through time ")


# In[63]:


#Creation of index column
data_final["timeIndex"] = pd.Series(np.arange(len(data_final['issue_mean'])), index=data_final.index)

data_final.head(3)


# In[64]:


# Categorical variables "months" and "weekday" transformed into dummy variables
dummies_mes = pd.get_dummies(data_final['months'], drop_first=True, prefix='Month')
data_final_1 = data_final.join(dummies_mes)
dummies_weekday = pd.get_dummies(data_final['weekday'], drop_first=True, prefix='weekday')
data_final_2 = data_final_1.join(dummies_weekday)
data_final_3 = data_final_2.drop(['months','weekday'],axis=1)
data_final_3


# In[65]:


# train test split
# (shuffle=False to avoid mixing observations, to have continuity between train and test sets)
df_train, df_test = train_test_split(data_final_3, test_size=0.3, random_state=1986, shuffle=False)
df_train.tail()


# In[66]:


df_test.head()


# In[67]:


# Linear Regression With Time Series
# Dependent variables: months
model_lineal = smf.ols('issue_mean ~ timeIndex + Month_Dec + Month_Feb + Month_Jan +  Month_Mar + Month_May + Month_Nov + Month_Oct + Month_Sep ' ,data = df_train).fit()

model_lineal.summary()


# #### Good performance. R2 = 91%

# #### Linear regression

# In[68]:


# Linear Regression With Time Series
# Dependent variables: weekday
model_lineal_2 = smf.ols('issue_mean ~ timeIndex + weekday_Monday + weekday_Saturday + weekday_Sunday + weekday_Thursday + weekday_Tuesday + weekday_Wednesday' ,data = df_train).fit()
model_lineal_2.summary()


# #### Poor performance. R2 = 41%. 

# In[69]:


# Predictions of "model_lineal"  (lineal model with time series, dependent variables: months), over train and test tests
df_train['model_lineal_est'] = model_lineal.predict(df_train[['timeIndex', 'Month_Dec' , 'Month_Feb' , 'Month_Jan' , 'Month_Mar' , 'Month_May' , 'Month_Nov' ,'Month_Oct' , 'Month_Sep']])


df_test['model_lineal_est'] = model_lineal.predict(df_test[['timeIndex', 'Month_Dec' , 'Month_Feb' , 'Month_Jan' , 'Month_Mar' , 'Month_May' , 'Month_Nov' ,'Month_Oct' , 'Month_Sep']])


# In[70]:


# Plot predictions over train set
# issue mean: real value
# model_lineal_est: predictions
df_train.plot(kind = "line", x = "month_year", y = ['issue_mean', 'model_lineal_est']);


# In[71]:


# Plot predictions over test set
# issue mean: real value
# model_lineal_est: predictions
df_test.plot(kind = "line", x = "month_year", y = ['issue_mean', 'model_lineal_est']);


# In[72]:


# Performance analyse with RMSE (root mean square error, compares predicted values with observed values)
# function to calculate RMSE
def RMSE(predicted, actual):
    mse = (predicted - actual) ** 2
    rmse = np.sqrt(mse.sum() / mse.count())
    return rmse


# In[73]:


#Metrics
df_Results = pd.DataFrame(columns = ["Model", "RMSE"])
df_Results.loc[0, "Model"] = "Lineal"
df_Results.loc[0, "RMSE"] = RMSE(df_test.model_lineal_est, df_test.issue_mean)
df_Results


# #### ARIMA model (AutoRegressive Integrated Moving Average model)

# In[74]:


# residue: differences between original values and predicted values by the model with back transformation.

residue = df_train['issue_mean'] - df_train['model_lineal_est']

plt.plot(df_train.timeIndex, residue, '-');


# In[75]:


# Application of Augmented Dickey-Fuller test: 
# tests the null hypothesis that a unit root is present in an autoregressive (AR) time series model. 
result = adfuller(residue)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in  result[4].items():
    print('Critic value %s: %.2f' % (key,value))


# p-value (0.072717) > 0.05 , fail to reject 𝐻0 , the series is not stationary.

# To determine p and q parameters, we need to generate ACF and PACF graphics

# In[76]:


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """ 
        Plot time series, ACF (Autocorrelation Function), PACF (Partial Autocorrelation Function) y Augmented Dickey-Fuller test
        
        y - time series
        lags -  fixed amount of passing time
        
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        
        # axes
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        
        # Augmented Dickey-Fuller test p-value
        p_value = sm.tsa.stattools.adfuller(y)[1]
        
        ts_ax.set_title('Time series analysis\n Dickey-Fuller: p={0:.5f}'                        .format(p_value))
        
        # ACF plot
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        # PACF plot
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[77]:


import statsmodels.api as sm
tsplot(residue, lags=2)


# In[78]:


#ARIMA model(p,d,q) as per graphic results.
model_ARIMA = sm.tsa.arima.ARIMA(residue, order=(1,0,1))
results_ARIMA = model_ARIMA.fit()
results_ARIMA.fittedvalues.head(36)
print(results_ARIMA.summary())

#akaike information criterion (AIC) & Schwarz information criterion (SIC): lower value, better model


# In[79]:


#Graphic of residues and ARIMA model
plt.figure(figsize=(7,3.5))
residue.plot()
results_ARIMA.fittedvalues.plot();


# In[80]:


results_ARIMA.fittedvalues.head()


# In[81]:


#ARIMA residues
res_ARIMA =  results_ARIMA.fittedvalues - residue


# In[82]:


#plot ARIMA residues
tsplot(res_ARIMA, lags=2)


# In[83]:


#ARIMA predictions
predictions_ARIMA= results_ARIMA.forecast(len(df_test['issue_mean'])+2, alpha=0.05)


# In[84]:


df_train['lineal_model_ARIMA'] = df_train['model_lineal_est'] + results_ARIMA.fittedvalues

df_test['lineal_model_ARIMA'] = df_test['model_lineal_est'] + predictions_ARIMA


# In[85]:


#Comparison of train model estimations
df_train.plot(kind = "line", y = ['issue_mean', 'lineal_model_ARIMA']);


# In[86]:


#Comparison of test model estimations
df_test.plot(kind = "line", y = ['issue_mean', 'lineal_model_ARIMA']);


# In[87]:


#Root mean square error(RMSE) of  Linear model and Lineal model + ARIMA
df_Results.loc[1, "Model"] = "Lineal Model + ARIMA"
df_Results.loc[1, "RMSE"] = RMSE(df_test['lineal_model_ARIMA'], df_test['issue_mean'])
df_Results


# ### The Linear model (lineal model with time series with months as dependent variables), has a similar RMSE in comparison to Lineal model + ARIMA. 
# ### Linear model will be combined with K Neighbors classifier 

# In[88]:


# Collection of probabilities obtained with time series models
data_models_series = df_train.append(df_test)
data_models_series


# # Models - Part 3 - Mixed model
# ### Combination of probabilities of The lineal model with time series and months as dependent variables, and K Neighbors classifier 

# In[89]:


# Join of K Neighbors classifier predictions with X_test_kbest
y_proba = pd.DataFrame(y_preds_knn_proba)
X_test_mix_1 = X_test_kbest.join(y_proba)
X_test_mix_2 = X_test_mix_1.reset_index()
X_test_mix_2


# In[90]:


#X_test index
index = pd.DataFrame(X_test.index)
index_1 = index.reset_index()
index_1


# In[91]:


X_test_mix_3= pd.merge(X_test_mix_2, index_1, on= 'index')
X_test_mix_4 = X_test_mix_3.drop(['index'],axis=1)
X_test_mix_5 = X_test_mix_4.rename(columns={'0_x':'proba_0',
                                   1:'proba_1','0_y':'index'})

# date test with K neighbours classifier probabilities
X_test_mix_5


# In[92]:


data_date = pd.DataFrame(data.date)
data_date_index = data_date.reset_index()
X_test_index = X_test.reset_index()
X_test_dd = pd.merge(X_test_index,data_date_index, on = 'index')
X_test_dd


# In[93]:


# Dates and index obtained
X_test_date_1 = pd.DataFrame(X_test_dd.loc[:,['index','date']])

X_test_date_1


# In[94]:


# creation of new variable "month_year"
# data joined with probabilities obtained with time series models
X_test_date_1['date'] = pd.to_datetime(X_test_date_1['date'], format='%Y/%m/%d')
X_test_date_1['Year'] = X_test_date_1['date'].dt.year 
X_test_date_1['Month'] = X_test_date_1['date'].dt.month 
X_test_date_1['year_str'] = X_test_date_1.Year.astype(str)
X_test_date_1['month_str'] = X_test_date_1.Month.astype(str)
X_test_date_1['month_year'] = X_test_date_1.month_str.str.cat(X_test_date_1.year_str, sep='/')

X_test_date_1['month_year'] = pd.to_datetime(X_test_date_1['month_year'], format='%m/%Y')

X_test_date_2= X_test_date_1.drop(['date','Year','Month','year_str','month_str'],axis=1)
data_models_series_2 = data_models_series.reset_index()

X_test_date_3 =pd.merge(X_test_date_2,data_models_series, on = 'month_year')
X_test_date_3


# In[95]:


#deletion of unncessary columns 
X_test_date_4 = X_test_date_3.drop([ 'issue_x', 'issue_mean', 'timeIndex', 'Month_Dec',
       'Month_Feb', 'Month_Jan', 'Month_Jun', 'Month_Mar', 'Month_May',
       'Month_Nov', 'Month_Oct', 'Month_Sep', 'weekday_Monday',
       'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
       'weekday_Tuesday', 'weekday_Wednesday'],axis=1)
X_test_date_4


# In[96]:


# Data test joined with probabilities obtained with time series models and K neighbours classifier
X_test_mixed = pd.merge(X_test_mix_5,X_test_date_4, on = 'index')
# Combined probability of issue=1 by K neighbours classifier and probabilities of linear time serie model
X_test_mixed['mixed_prob'] = X_test_mixed['proba_1'] * X_test_mixed['model_lineal_est']
# data ordered per date
X_test_mixed_final = X_test_mixed.sort_values(by='month_year')
X_test_mixed_final


# In[97]:


# Best treshold found to optimize model metrics: 0.025
# if probability ("mixed_prob") is >= to 0.025, issue= 1, if not issue= 0.
threshold = 0.025
X_test_mixed_final['issue_prob_final'] =  X_test_mixed_final.apply(lambda x: 1 if (x['mixed_prob']>= threshold) else 0,axis=1)
X_test_mixed_final


# In[98]:


# histogram of "mixed_prob"
hist = X_test_mixed_final.mixed_prob.hist(bins=30)


# In[99]:


# Final issue predictions of K neighbours classifier and linear time serie model combined
y_preds_knn_lineal_mixed= X_test_mixed_final.issue_prob_final
y_preds_knn_lineal_mixed


# In[100]:


def evaluate_model(y_test,y_pred,name):
    '''
    Function to calculate principal metrics to evaluate a classificator
    Input: y test and y predictions
    
    '''
    conf_mat = confusion_matrix(y_test, y_pred)
    print(name+ '\n\n')
    print('Confusion matrix\n')
    confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,fmt='4d')
    plt.ylabel('True')
    plt.xlabel('Predicted');
    
    print(classification_report(y_test,y_pred))
 
    return


# In[101]:


# Model metrics of K neighbour classifier
evaluate_model(y_test_last,y_preds_knn, 'K Neighbour classifier')


# In[102]:


## Model metrics of K neighbour classifier combined with linear time serie model
evaluate_model(y_test_last,y_preds_knn_lineal_mixed, 'K Neighbour Classifier + Linear Time serie model')


# ### K Neighbour Classifier combined with Linear Time serie model has better performance than K Neighbour Classifier alone

# In[ ]:




