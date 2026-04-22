import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#load data
diabetes_dataset = pd.read_csv('diabetes.csv')

# print first 5 raws
diabetes_dataset.head()

# No of raws & columns
diabetes_dataset.shape

# display statistical measures
diabetes_dataset.describe()

#counts how many times each unique value appears in label column
diabetes_dataset['Outcome'].value_counts()

#mean of each numeric column
diabetes_dataset.groupby('Outcome').mean()

# separate data & lable
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

#Data Standardization
scaler = StandardScaler()
scaler.fit(X)
stded_data = scaler.transform(X) # can do as stded_data = scaler.fit_transform(X)
print(stded_data)

X = stded_data
#Y = diabetes_dataset['Outcome'] #not necessasry

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# traiining the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# model evaluation -> accuracy on traning data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of traning data: ', training_data_accuracy)

# model evaluation -> accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of test data: ', test_data_accuracy)

# make predictive system
#input_data = (1,85,66,29,0,26.6,0.351,31) # no diabetes data

input_data = (0,137,40,35,168,43.1,2.288,33) # diabetes data

# input_data -> numpy array
input_data_array = np.asarray(input_data)

# reshape the data, , as only predicting for one instance
input_data_reshaped = input_data_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
