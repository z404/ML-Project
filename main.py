print('Script to automate increase in accuracy of the ML classifier model\nImporting packages...\n')
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

##IMPORTING DIFFERENT CLASSIIFIERS
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

#Reading csv and converting all entries to number
file_name = input("Enter file name of train csv: ")
dataset = pd.read_csv(file_name,header=0, index_col = 's.no',parse_dates=True)
predict_label = input('Enter the label to be predicted: ')

##Creating the train data and test data
if predict_label in dataset.keys():
    print('Proceeding...')
else:
    print('Error, label doesn\'nt exist in the dataset')
labels = [i for i in dataset.keys() if i != predict_label]
X = dataset[labels]
y = dataset[predict_label]

#creating list of objects of all classifiers in sklearn
classifier_list = []
classifier_name = ['Logistic Regression','Random Forest','AdaBoost Classifier']
classifier_list.append(LogisticRegression())
classifier_list.append(RandomForestClassifier())
classifier_list.append(AdaBoostClassifier())

#running with full dataset first
index = 0
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
for clf in classifier_list:
    print('Running Full Dataset with model number',index,':',classifier_name[index])
    trained_clf = clf.fit(X_train,y_train)
    print('Done training, predicted accuracy:',end=' ')
    print(clf.score(X_test,y_test)*100)
    
