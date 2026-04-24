import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# load dataset
loan_data = pd.read_csv('loan_data.csv')

# print first 5 raws
loan_data.head()

# number of raws and columns
loan_data.shape

# statistical measure
loan_data.describe()

# number of missing value
loan_data.isnull().sum()

# drop missing valuwa
loan_data = loan_data.dropna()
loan_data.isnull().sum()

# lable encoding
loan_data.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace=True)

loan_data.head()

# dependent column values
loan_data['Dependents'].value_counts()

# replace 3+ value to 4
loan_data = loan_data.replace(to_replace='3+', value=4)

# dependednt values
loan_data['Dependents'].value_counts()

# education & loan status
sns.countplot(x='Education', hue='Loan_Status', data=loan_data)

# marital status & loan status
sns.countplot(x='Married', hue='Loan_Status', data=loan_data)

# convert categorical -> numerical
loan_data.replace({'Married':{'No':0, 'Yes':1}, 'Gender':{'Male':1, 'Female':0}, 'Self_Employed':{'No':0, 'Yes':1},
                   'Property_Area':{'Rural':0, 'Semiurban':1, 'Urban':2}, 'Education':{'Graduate':1, 'Not Graduate':0}}, inplace=True)

loan_data.head()

# split data and lable
X = loan_data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_data['Loan_Status']

print(X)
print(Y)

# split -> train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# train model -> Support Vector Machine Model
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)

# accuracy -> training data
X_train_prediction = svm_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy -> testing data
X_test_prediction = svm_classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the testing data : ', testing_data_accuracy)