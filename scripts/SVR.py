import numpy as np

def instanceGenerator(num_instances, file_loc, y_index, seperator=";", skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)
            X = np.array(instance_data_raw[:y_index])
            y = instance_data_raw[y_index]

            yield X, y

for X, y in instanceGenerator(20, "../WineData/winequality-red.csv", 11):
    print(X, y)