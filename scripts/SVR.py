import numpy as np
from math import ceil

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Assumed input is a CSV file of numeric data
def instanceGenerator(num_instances, file_loc, seperator=";", skip_instances = 0, skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        #print(input_file.readline()[skip_instances:])
        for i in range(skip_instances):
            input_file.readline()
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)

            yield np.array(instance_data_raw, dtype=np.float64)

data_size = 4898

data = np.array([instance for instance in instanceGenerator(data_size, "../WineData/winequality-white.csv")])

#X_data = np.array([np.array([data[i][10], data[i][5], data[i][2]]) for i in range(len(data))])
X_data = np.array([data[i][:11] for i in range(len(data))])
y_data = np.array([data[i][11] for i in range(len(data))])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=6)

#X_train = np.array([training_data[i][:11] for i in range(len(training_data))])
#y_train = np.array([training_data[i][11] for i in range(len(training_data))])

#X_test = np.array([training_data[i][:11] for i in range(len(testing_data))])
#y_test = np.array([training_data[i][11] for i in range(len(testing_data))])


scaler = StandardScaler().fit(X_train)
clf = SVR(kernel="rbf", C=10, epsilon=0.25)
clf.fit(scaler.transform(X_train), y_train)
pred = clf.predict(scaler.transform(X_test))

print("With float output")
print("R2 score:\t\t", r2_score(y_test, pred))
print("Explained Var:\t\t", explained_variance_score(y_test, pred))
print("Mean Absolute Error:\t", mean_absolute_error(y_test, pred))
print("Mean Squared Error:\t", mean_squared_error(y_test, pred))
print("Mean Squared Log Error:\t", mean_squared_log_error(y_test, pred))
print("Median Absolute Error:\t", median_absolute_error(y_test, pred))

pred_int = np.array([round(y) for y in pred], dtype=np.int32)
y_test_int = np.array([round(y) for y in y_test], dtype=np.int32)

print("With int output")
print("R2 score:\t\t", r2_score(y_test_int, pred_int))
print("Explained Var:\t\t", explained_variance_score(y_test_int, pred_int))
print("Mean Absolute Error:\t", mean_absolute_error(y_test_int, pred_int))
print("Mean Squared Error:\t", mean_squared_error(y_test_int, pred_int))
print("Mean Squared Log Error:\t", mean_squared_log_error(y_test_int, pred_int))
print("Median Absolute Error:\t", median_absolute_error(y_test_int, pred_int))
