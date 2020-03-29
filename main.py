print('Script to automate increase in accuracy of the ML classifier model\nImporting packages...\n')
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import os
import copy

##IMPORTING DIFFERENT CLASSIIFIERS
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.simplefilter("ignore")

#Reading csv
file_name = input("Enter file name of train csv: ")
dataset = pd.read_csv(file_name,header=0, index_col = 's.no',parse_dates=True)
predict_label = input('Enter the label to be predicted: ')


##Creating the train data and test data
if predict_label in dataset.columns:
    print('Proceeding...')
else:
    print('Error, label doesn\'t exist in the dataset')
labels = [i for i in dataset.keys() if i != predict_label]
X = dataset[labels]
y = dataset[predict_label]

total = 1
for i in range(len(X.columns)):
    total+=1
for i in range(len(X.columns)):
    for j in range(i,len(X.columns)):
        if i != j:
            total+=1
total *= 7  #7 algorithms
steps = 0

#creating list of objects of all classifiers in sklearn
classifier_list = []
classifier_name = ['Logistic Regression','Random Forest','AdaBoost Classifier','Gaussian Process Classifier','Decision Tree Classifier','Quadratic Discriminant Analysis','SVC']
classifier_list.append(LogisticRegression(C=4, penalty='l1'))
classifier_list.append(RandomForestClassifier(n_estimators = 1000,min_samples_leaf = 80))
classifier_list.append(AdaBoostClassifier())
classifier_list.append(GaussianProcessClassifier())
classifier_list.append(DecisionTreeClassifier())
classifier_list.append(QuadraticDiscriminantAnalysis())
classifier_list.append(SVC())

#defining accuracy logger
accuracy_log = []
desc_log = []

#running with full dataset first
max_acc = 0
desc = ''
index = 0
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

for clf in classifier_list:
    print('Current Max:',max_acc)
    print('Current Max algorithm:',desc)
    steps+=1
    print(steps,'/',total)
    print('Running Full Dataset with model number',index+1,':',classifier_name[index])
    trained_clf = clf.fit(X_train,y_train)
    print('Done training, predicted accuracy:',end=' ')
    score = clf.score(X_test,y_test)*100
    if score > max_acc:
        max_acc = score
        desc = classifier_name[index]+' / full dataset'
    print(score)
    print()
    os.system('clear')
    index+=1

print('Maximum accuracy achived =',max_acc)
print('Algorithm used:',desc)
accuracy_log.append(max_acc)
desc_log.append(desc)
print('###########################################################################')

#dropping single column and recieving accuracy
drop_1D = X_train.columns
for i in drop_1D:
    new_X_train = X_train.drop([i],axis=1)
    new_X_test = X_test.drop([i],axis=1)
    index = 0
    for clf in classifier_list:
        print('Current Max:',max_acc)
        print('Current Max algorithm:',desc)
        steps+=1
        print(steps,'/',total)
        print('Dropping column',i,'and running classifier',index+1,':',classifier_name[index])
        trained_clf = clf.fit(new_X_train,y_train)
        print('Done training, predicted accuracy:',end=' ')
        score = clf.score(new_X_test,y_test)*100
        if score > max_acc:
            max_acc = score
            desc = classifier_name[index]+' / dropped column '+i
        print(score)
        print()
        os.system('clear')
        index+=1
        
print('Maximum accuracy achived =',max_acc)
print('Algorithm used:',desc)
accuracy_log.append(max_acc)
desc_log.append(desc)
print('###########################################################################')

#dropping two columns and recieving accuracy
for i in range(len(drop_1D)):
    for j in range(i,len(drop_1D)):
        if i != j:
            new_X_train = X_train.drop([drop_1D[i],drop_1D[j]],axis=1)
            new_X_test = X_test.drop([drop_1D[i],drop_1D[j]],axis=1)
            index = 0
            for clf in classifier_list:
                print('Current Max:',max_acc)
                print('Current Max algorithm:',desc)
                steps+=1
                print(steps,'/',total)
                print('Dropping column',drop_1D[i],'and',drop_1D[j],'and running classifier',index+1,':',classifier_name[index])
                trained_clf = clf.fit(new_X_train,y_train)
                print('Done training, predicted accuracy:',end=' ')
                score = clf.score(new_X_test,y_test)*100
                if score > max_acc:
                    max_acc = score
                    desc = classifier_name[index]+' / dropped column '+drop_1D[i]+' and '+drop_1D[j]
                print(score)
                print()
                os.system('clear')
                index+=1

print('Maximum accuracy achived =',max_acc)
print('Algorithm used:',desc)
accuracy_log.append(max_acc)
desc_log.append(desc)
print('###########################################################################')

print(accuracy_log)
print(desc_log)
