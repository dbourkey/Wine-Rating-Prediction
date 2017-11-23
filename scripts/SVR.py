import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

# Assumed input is a CSV file of numeric data
def instanceGenerator(num_instances, file_loc, seperator=";", skip_instances = 0, skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        input_file.readline()[0:]
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)

            yield np.array(instance_data_raw, dtype=np.float64)

train = 1500
test = 400

training_data = np.array([instance for instance in instanceGenerator(train, "../WineData/winequality-red.csv")])
testing_data = np.array([instance for instance in instanceGenerator(test, "../WineData/winequality-red.csv", skip_instances=train)])

X_train = np.array([training_data[i][:11] for i in range(len(training_data))])
y_train = np.array([training_data[i][11] for i in range(len(training_data))])

X_test = np.array([training_data[i][:11] for i in range(len(testing_data))])
y_test = np.array([training_data[i][11] for i in range(len(testing_data))])

scaler = StandardScaler().fit(X_train)


clf = SVR(kernel="rbf", C=800, epsilon=0.0625)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("R2 score:\t\t", r2_score(y_test, pred))
print("Explained Var:\t\t", explained_variance_score(y_test, pred))
print("Mean Absolute Error:\t", mean_absolute_error(y_test, pred))
print("Mean Squared Error:\t", mean_squared_error(y_test, pred))
print("Mean Squared Log Error:\t", mean_squared_log_error(y_test, pred))
print("Median Absolute Error:\t", median_absolute_error(y_test, pred))