# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_Train, y_test = train_test_split(X, y, test_size=1/3,random_state=0)

#Feature scaling
'''

'''
# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(X_train, y_Train)

# predicting test set results

y_pred = regresser.predict(X_test)

# visualizing training test result
plt.scatter(X_train, y_Train, color='red')
plt.plot(X_train, regresser.predict(X_train),    color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()