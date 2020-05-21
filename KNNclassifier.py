import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

#Import DataSet and drop unused Columns
dataset = pd.read_csv("Tickets2014.csv")
dataset = dataset.drop("ViolationChargedCode",axis=1)
dataset = dataset.drop("ViolationYear",axis=1)

#Deal with missing data: Use fillna or dropna() or imputer version from Udemy
dataset = dataset.dropna()

# Split into features and targets
targets = dataset.iloc[:,:1]
features = dataset.iloc[:, dataset.columns != 'ViolationDescription']

#Deal with categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#LabelEncode categorical data
DayLabelEncoder = LabelEncoder()
features['ViolationDayofWeek'] = DayLabelEncoder.fit_transform(features['ViolationDayofWeek'])
genderEncoder = LabelEncoder()
features['Gender'] = genderEncoder.fit_transform(features['Gender'])
#Onehotencoding for Days of week cuz there are more than 2 options
features = pd.concat([features,pd.get_dummies(features['ViolationDayofWeek'], prefix='day')],axis=1)
features = features.drop('ViolationDayofWeek',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_test)
print("The accuracy using KNN is: ")
print(accuracy_score(y_test,y_pred))










