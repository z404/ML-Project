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

#creating list of objects of all classifiers in sklearn
classifier_list = []
classifier_name = ['Logistic Regression','Random Forest','AdaBoost Classifier','Gaussian Process Classifier','Decision Tree Classifier','Quadratic Discriminant Analysis','SVC']
classifier_list.append(LogisticRegression(C=4, penalty='l1'))
classifier_list.append(RandomForestClassifier(n_estimators = 200,min_samples_leaf = 80))
classifier_list.append(AdaBoostClassifier())
classifier_list.append(GaussianProcessClassifier())
classifier_list.append(DecisionTreeClassifier())
classifier_list.append(QuadraticDiscriminantAnalysis())
classifier_list.append(SVC())

#defining accuracy logger
accuracy_log = []
desc_log = []

#running accuracy model
max_acc = 0
alg_desc = ''
col_desc = ''
index = 0
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

list_of_five = X_train.columns
swap_list_list = [[]]
for swap in list_of_five:
    temp_lists = []
    for list_stub in swap_list_list:
        this_list = copy.copy(list_stub)
        this_list.append(swap)
        temp_lists.append(this_list)
        temp_lists.append(list_stub)
    swap_list_list = temp_lists
total = len(swap_list_list)
counter = 0

##temp_list = []
##for i in swap_list_list:
##    temp_list.append(len(i))
##for i in range(len(temp_list)):
##    for j in range(0,len(temp_list)-i-1):
##        if temp_list[j]>temp_list[j+1]:
##            temp_list[j],temp_list[j+1] = temp_list[j+1],temp_list[j]
##            swap_list_list[j],swap_list_list[j+1] = swap_list_list[j+1],swap_list_list[j]

swap_list_list.sort()

for i in swap_list_list:
    if len(i) != len(X_train.columns):
        new_X_train = X_train.drop(i,axis=1)
        new_X_test = X_test.drop(i,axis=1)
        index = 0
        for clf in classifier_list:
            print('Current Maximum accuracy reached:',max_acc)
            print('Algorithm used to obtain maximum accuracy:',alg_desc)
            print('Columns dropped to obtain maximum accuracy:',col_desc)
            counter+=1
            print(counter,'/',total)
            print('Dropping column',i,'and running classifier',index+1,':',classifier_name[index])
            trained_clf = clf.fit(new_X_train,y_train)
            score = clf.score(new_X_test,y_test)*100
            if score >= max_acc:
                max_acc = score
                alg_desc = classifier_name[index]
                col_desc = i
            os.system('clear')
            index+=1

os.system('clear')
print('Maximum accuracy reached:',max_acc)
print('Algorithm used to obtain accuracy:',alg_desc)
print('Columns dropped to obtain accuracy:',col_desc)
