import numpy as np

from sklearn import svm

# Assumed input is a CSV file of numeric data
def instanceGenerator(num_instances, file_loc, seperator=";", skip_instances = 0, skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        input_file.readline()[0:]
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)

            yield np.array(instance_data_raw, dtype=np.float64)

train = 8
test = 2
"""
training_data = [(X_instance, y_instance) for X_instance, y_instance in 
    instanceGenerator(train, "../WineData/winequality-red.csv", 11)][0]
testing_data = [(X_instance, y_instance) for X_instance, y_instance in 
    instanceGenerator(test, "../WineData/winequality-red.csv", 11, skip_instances=train)]
"""

training_data = np.array([instance for instance in instanceGenerator(train, "../WineData/winequality-red.csv")])
testing_data = np.array([instance for instance in instanceGenerator(test, "../WineData/winequality-red.csv", skip_instances=train)])

X_train = np.array([training_data[i][:11] for i in range(len(training_data))])
y_train = np.array([training_data[i][11] for i in range(len(training_data))])

X_train = np.array([training_data[i][:11] for i in range(len(testing_data))])
y_train = np.array([training_data[i][11] for i in range(len(testing_data))])