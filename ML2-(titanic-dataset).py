# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:36:16 2018

@author: Shashwat Swain
"""
# Import the needed references
import pandas as pd
import numpy as np
import csv

# Import the necesserary models and model selection tools
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curves
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#import seaborn as sns
#Output plots in notebook
#%matplotlib inline

addpoly = True
plot_lc = 0     # 1--display learning curve/0 -- don't display

#Loading the data sets from the csv files
print('---------load train & test file--------')
train_dataset = pd.read_csv('dataset/train.csv')
test_dataset = pd.read_csv('dataset/test.csv')
print('train dataset: %s, test dataset %s' %(str(train_dataset.shape),str(test_dataset.shape)))

#Checking if there are missing values in the dataset
datasetHasNan=False
if train_dataset.count().min() == train_dataset.shape[0]:
    print(">>There are NO missing values in the data")
else:
    datasetHasNan=True
    print(">>There are missing values in the dataset")   
    
#Checking if the "PassengerId"column has missing datas
if train_dataset.PassengerId.nunique() == train_dataset.shape[0]:
    print(">>No missing values in 'PassengerId' column")
else:
    print(">>There are missing values in 'PassengerId' column")


print('------ train dataset column type information -----')
dtype_df = train_dataset.dtypes.reset_index()
dtype_df.columns = ["Count","Column Type"]
print(dtype_df.groupby("Column Type").aggregate("count").reset_index())

print('------- train dataset information -------')
print(dtype_df)

# Checking for missing values & listing them
print('--- Checking for missing values and listing them ---')
if datasetHasNan == True:
    nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=["Train Dataset","Test Dataset"])
    print(nas)
    print("Nan in the data sets")
    print(nas[nas.sum(axis=1) > 0])

# Class vs Survived
print("Class vs Survived")
print(train_dataset[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by="Survived",ascending=False))

# Sex vs Survived
print("Sex vs Survived")
print(train_dataset[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# Sibsp vs Survived
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiances were ignored)
print("Sibsp vs Survived")
print(train_dataset[["SibSp","Survived"]].groupby("SibSp",as_index=False).mean().sort_values(by="Survived",ascending=False))

# Parch vs Survived
#Parent = mother, father
#Chile = daughter, son, stepbrother, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.
print("Parch vs Survived")
print(train_dataset[["Parch","Survived"]].groupby("Survived",as_index=False).mean().sort_values(by="Survived", ascending=False))

# Dataset Cleaning, fill NaN where needed and delete uneeded columns
print('----- Start data Cleaning -----')

#manage Age
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(), train_dataset["Age"].mean() + train_dataset["Age"].std(), size = train_dataset["Age"].isnull().sum())
test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(), test_dataset["Age"].mean() + test_dataset["Age"].std(), size = test_dataset["Age"].isnull().sum())

#print(train_random_ages)
train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages
#print(train_dataset["Age"])
train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age'] = test_dataset['Age'].astype(int)

#   Embarked
train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)

train_dataset['Port'] = train_dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S':0,'C':1,'Q':0}).astype(int)

del train_dataset['Embarked']
del test_dataset['Embarked']

#   Fare
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)

#Feature that tells whethers a passenger had a cabin on the titanic
train_dataset['Has_Cabin'] = train_dataset['Cabin'].apply(lambda x:0 if type(x) is float else 1)
test_dataset['Has_Cabin'] = test_dataset['Cabin'].apply(lambda x:0 if type(x) is float else 1)

#Complete clearning of Data
print("Dataset after being cleaned")
nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=["Train Dataset","Test Dataset"])
print(nas)

# engineer a new Title feature
print("----- Adding new features to the dataset -----")
# group them
full_dataset = [train_dataset, test_dataset]

## engineer the family size feature
for dataset in full_dataset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1

### new try
    
#   Create new feature IsAlone from FamilySize
for dataset in full_dataset:
    dataset['IsAlone'] = 0
    dataset.loc[dataset["FamilySize"]==1,"IsAlone"] = 1
    
############################################################
    
# Get titles from the names

train_dataset['Title'] = train_dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
test_dataset['Title'] = test_dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)

for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')#   necessary
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')#   necessary
    dataset['Title'] = dataset['Title'].replace('Mlle','Mr')#   necessary

## Create a new column "FamilySizeGroup" and assign "Alone", "Small" and "Big"
for dataset in full_dataset:
    dataset["FamilySizeGroup"] = "Small"
    dataset.loc[dataset["FamilySize"]==1, "FamilySizeGroup"] = "Alone"
    dataset.loc[dataset["FamilySize"]>=5, "FamilySizeGroup"] = "Big"

## Get the average survival rate of different familySizes
print(train_dataset[['FamilySize','Survived']].groupby(["FamilySize"], as_index=False).mean().sort_values(by="Survived",ascending=False))

for dataset in full_dataset:
    dataset["Sex"] = dataset["Sex"].map({'male':0,'female':1}).astype(int)

# Making groups by Age
for dataset in full_dataset:
    dataset.loc[ dataset["Age"]<=14, "Age"] = 0
    dataset.loc[ (dataset["Age"]>14)&(dataset["Age"]<=32), "Age"] = 1
    dataset.loc[ (dataset["Age"]>32)&(dataset["Age"]<=48), "Age"] = 2
    dataset.loc[ (dataset["Age"]>48)&(dataset["Age"]<=64), "Age"] = 3
    dataset.loc[ dataset["Age"]>64, "Age"] = 4
    
#    
for dataset in full_dataset:
    dataset.loc[ dataset['Fare']<7.91, "Fare"] = 0
    dataset.loc[ (dataset['Fare']>7.91)&(dataset['Fare']<=14.454), 'Fare'] = 1
    dataset.loc[ (dataset['Fare']>14.454)&(dataset['Fare']<=31), 'Fare'] = 2
    dataset.loc[ dataset['Fare']>31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#map the new features

title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
family_mapping = {"Small":0, "Alone":1, "Big":2}

print('Mapping the "Title" and "FamilySizeGroup" into numerical data(before)')

print("train_dataset.Title")
print(train_dataset['Title'].head())
print("train_dataset.FamilySizeGroup")
print(train_dataset['FamilySizeGroup'].head())
#       mapping 
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

print('Mapping the "Title" and "FamilySizeGroup" into numerical data(after)')
print("train_dataset.Title")
print(train_dataset['Title'].head())
print("train_dataset.FamilySizeGroup")
print(train_dataset['FamilySizeGroup'].head())
    
    
    
# engineer a new feature

for dataset in full_dataset:
    dataset['IsChildandRich'] = 0
    dataset.loc[((dataset['Age']) <= 0)&(dataset["Pclass"] == 1),"IsChildandRich"] = 1
    dataset.loc[((dataset['Age']) <= 0)&(dataset["Pclass"] == 1),"IsChildandRich"] = 1

##   Additional features
## Age*Class
#for dataset in full_dataset:
#    dataset['Age*Class'] = dataset.Age*dataset.Pclass
    
## Sex*Class
#for dataset in full_dataset:
#    dataset['Sex*Class'] = dataset.Sex*dataset.Pclass

## Sex*Age
#for dataset in full_dataset:
#    dataset['Sex*Age'] = dataset.Sex*dataset.Age

## Age*Class*Sex
#for dataset in full_dataset:
#    dataset['Age*Class*Sex'] =   (dataset.Age*dataset.Pclass) + dataset.Sex  
    

#for data in full_dataset:
    #classify cabin by fare
#    data['Cabin'] = data['Cabin'].fillna('X')
#    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
#    data['Cabin'] = data['Cabin'].replace(['A','D','E','T'],'M')
#    data['Cabin'] = data['Cabin'].replace(['B','C'],'H')
#    data['Cabin'] = data['Cabin'].replace(['F','G'],'L')
#    data['Cabin'] = data['Cabin'].map({'X':0,'L':1,'M':2,'H':3}).astype(int)

# We delete the 'Name','SibSp','Parch','FamilySize','Ticket', 'Port' Column from the datasets
# We don't need it in the analysis

del train_dataset['Name']
del test_dataset['Name']

del train_dataset['Parch']
del test_dataset['Parch']

del train_dataset['SibSp']
del test_dataset['SibSp']

del train_dataset['FamilySize']
del test_dataset['FamilySize']

del train_dataset['Cabin']
del test_dataset['Cabin']

del train_dataset['Ticket']
del test_dataset['Ticket']

del train_dataset['Port']
del test_dataset['Port']


print(' ---- Finish data cleaning ---- ')
print('train dataset: %s, test dataset: %s' %(str(train_dataset.shape),str(train_dataset.shape))    )

#deleting PassengerId
del train_dataset['PassengerId']

X_train = train_dataset.drop("Survived",axis=1)     #input features
Y_train = train_dataset["Survived"]     #target value
X_test = test_dataset.drop("PassengerId",axis=1).copy()

print("X_train.shape :%s"%str(X_train.shape))
print("Y_train.shape :%s"%str(Y_train.shape))
print("X_test.shape :%s"%str(X_test.shape))

#   try Polynomials
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

if addpoly:
    all_data = pd.concat((X_train,X_test),ignore_index=True)    #ignore_index=True concats both the datasets vertically
    
    scaler = MinMaxScaler()
    scaler.fit(all_data)    
    all_data = scaler.transform(all_data)
    
    poly = PolynomialFeatures(2)
    all_data = poly.fit_transform(all_data)
    
    X_train = all_data[:train_dataset.shape[0]]
    X_test = all_data[train_dataset.shape[0]:]
    
    ##
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)

#   learning Curve
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0) #For cross-validation of the training set   
logreg_model = LogisticRegression() #Using Logistic Regression

def Learning_curve_model(X, Y, model, cv, train_sizes):
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, train_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

#Learn curve
lc=0
if lc==1:
    train_size=np.linspace(.1,1,15)
    Learning_curve_model(X_train, Y_train, logreg_model, cv, train_size)

#   ===Logistic Regression===
print("////////////////////////")
print("...Using Logistic Regression...")

logreg = LogisticRegression()   #(C=0.1, penalty='l1', tot=1e-6)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

result_train = logreg.score(X_train, Y_train)
result_val = cross_val_score(logreg, X_train, Y_train, cv=6).mean()
print('training score = %s, while validation score %s' %(result_train, result_val))


#   Support Vector Machine
print("////////////////////////")
print("...Using Support Vector Machine...")    

svc = SVC(C = 0.1, gamma = 0.1)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

result_train = svc.score(X_train, Y_train)
result_val = cross_val_score(svc, X_train, Y_train, cv=5).mean()
print("Training score = %s, while validation score = %s" %(result_train, result_val))

#   Random Forests
print("////////////////////////")
print("...Using Random Forests Classifier...")

random_forest = RandomForestClassifier(criterion='gini', n_estimators=1000, min_samples_split=10, min_samples_leaf=1, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

seed = 42
random_forest=RandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='auto', oob_score=False, random_state=seed, n_jobs=1, verbose=0, bootstrap=False)

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

result_train = random_forest.score(X_train, Y_train)
result_val = cross_val_score(random_forest, X_train, Y_train, cv=5).mean()

print('Training score = %s, while validation score = %s' %(result_train, result_val))

final_prediction = pd.DataFrame({"PassengerId": test_dataset['PassengerId'], "Survived": Y_pred})
final_prediction.to_csv('dataset/titanic.csv', index=False)
print(final_prediction)



