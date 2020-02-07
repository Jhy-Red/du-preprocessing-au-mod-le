#!/usr/bin/python3

# Imports 
from pandas import isna, unique, isnull, read_csv, value_counts,get_dummies,concat
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as reglog
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Dataset 
data_list = "age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hour-per-week","native-country","Salary"
adult,adult_test = read_csv('Data/adult.data', header=None, sep = ' *, *', names = data_list,engine = 'python' ), read_csv('Data/adult.test', sep = ' *, *', names = data_list, engine = 'python')
#region Analyse du set
adult.isna().sum() 
adult_test.isna().sum()
adult.isnull().sum()
adult_test.isnull().sum()
# Visual

plt.figure(figsize = (12,8))
sns.heatmap(adult.isnull(),yticklabels=False,cbar=True,cmap='plasma')

#
how_many_unknow = adult.loc[np.where(adult == '?')[0]] # 2419 total raws

adult.loc[np.where(adult["native-country"] == '?')[0]] # native country # 583 row # recommendation drop

adult.loc[np.where(adult["occupation"] == '?')[0]] # 1843 rows -- occupation 7 sup
adult.loc[np.where(adult["workclass"] == '?')[0]] # 1836 rows
adult.loc[(adult["workclass"] == '?')&(adult["occupation"] == '?')] ## 1836 rows # alternative but to heavy : checking_job = adult.groupby(['workclass','occupation']) ## too much line have to use head()
### to justify 4262 raws with 1836 combined rows. total loss 2426 raws

status_hours = adult['hour-per-week'].describe()

# echantillon test :  963 rows - workclass, 966 Occupation # native coutry 274

#endregion
# Update table
adult_test = adult_test.drop([0], axis = 0)
adult.replace('?', np.nan, inplace = True) ; adult_test.replace('?',np.nan, inplace = True)
adult_test["Salary"] = adult_test["Salary"].str.replace(".","") # Alternative : adult_test['Salary'].replace('<=50K.',"<=50K", inplace = True) ; adult_test['Salary'].replace('>50K.','>50K', inplace = True) 

# modification table : 
# Reunification de la variable capital
adult['Capital'] = (adult['capital-gain'] - adult['capital-loss'])
adult_test['Capital'] = (adult_test['capital-gain'] - adult_test['capital-loss'])


adult['gender_dum'] = get_dummies(adult["sex"],drop_first=True)
adult_test['gender_dum'] = get_dummies(adult_test["sex"],drop_first=True)
# Discretisation du secteur d'activités :

adult["Sector"] = ["Public" if x  == "State-gov" or x == "Federal-gov" or x == "Local-gov" else "Private" for x in adult["workclass"]]
adult_test["Sector"] = ["Public" if x  == "State-gov" or x == "Federal-gov" or x == "Local-gov" else "Private" for x in adult_test["workclass"]]

adult["Num_sector"] = [1 if x == "Public" else 0 for x in adult['Sector']]
adult_test["Num_sector"] =  [1 if x == "Public" else 0 for x in adult_test['Sector']]
# Train & Test

y_train = adult['Salary']
y_test = adult_test['Salary']

X_train = adult.drop(['native-country','capital-loss','capital-gain','education','Salary',"Sector"], axis = 'columns')
X_train = adult.drop(['native-country','capital-loss','capital-gain','workclass','education','marital-status','occupation','relationship','race','sex','Salary',"Sector"], axis = 'columns') # light_set

X_test = adult_test.drop(['native-country','capital-loss','capital-gain','education','Salary',"Sector"], axis = 'columns')
X_test = adult_test.drop(['native-country','capital-loss','capital-gain','workclass','education','marital-status','occupation','relationship','race','sex','Salary',"Sector"], axis = 'columns') # light_set

# Transformer les valeurs qualitatives en quantitatives

# Score Methode 1 
score_100= reglog(solver='lbfgs',multi_class='auto',penalty='none').fit(X_train,y_train).predict_proba(X_test)
score = reglog(solver='lbfgs',multi_class='auto',penalty='none').fit(X_train,y_train).predict(X_test)

print(classification_report(y_test, score))
print(accuracy_score(y_test, score))
print(confusion_matrix(y_test, score))


# Méthode 2

param_grid = {'n_neighbors': list(range(1,10))}

knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn.fit(X_train, y_train)
for mean, std, params in zip(knn.cv_results_['mean_test_score'],knn.cv_results_['std_test_score'],knn.cv_results_['params']):
    print("'accuracy' = {:.3f} (+/-{:.03f}) for {}".format(mean,std*2,params))

