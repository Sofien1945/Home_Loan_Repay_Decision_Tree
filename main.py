""" Home loan repay prediction using decision tree classifier
Day: 12.10.2021
Interpreter: Python 3.8
Done By: Sofien Abidi
Part of SIMPLEARN Machine Learning Course Example
"""
#Import Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = 'C:/Users/Sofien/Desktop/loan_data.csv'
balance_data = pd.read_csv(file_path)
balance_data.shape
#Seperate the Features and target
X = balance_data.iloc[:,1:5]
Y = balance_data.iloc[:,0]

#Splitting the test and train data
X_test, X_train, Y_test, Y_train = train_test_split(X, Y, test_size=0.3, random_state=100)

#Creation of model + trainig
model = DecisionTreeClassifier(criterion='entropy', random_state=100,max_depth=3,min_samples_leaf=5)
model.fit(X_train,Y_train)

#Model testing
y_pred = model.predict(X_test)

#Accuracy Verification
score = accuracy_score(Y_test,y_pred)
print("Score: ",score)