import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# Assumed input is a CSV file of numeric data
def instanceGenerator(num_instances, file_loc, seperator=";", skip_instances=0, skip_header=True):
    with open(file_loc, "r") as input_file:
        if skip_header:
            input_file.readline()
        for i in range(skip_instances):
            input_file.readline()
        for i in range(num_instances):
            instance_data_raw = input_file.readline().split(seperator)

            yield np.array(instance_data_raw, dtype=np.float64)

dataset = [item for item in instanceGenerator(4898, "../WineData/winequality-white.csv")]

fixed_acidity = np.array([instance[0] for instance in dataset])
volatile_acidity = np.array([instance[1] for instance in dataset])
citric_acid = np.array([instance[2] for instance in dataset])
residual_sugar = np.array([instance[3] for instance in dataset])
chlorides = np.array([instance[4] for instance in dataset])
free_sulfur_dioxide = np.array([instance[5] for instance in dataset])
total_sulfur_dioxide = np.array([instance[6] for instance in dataset])
density = np.array([instance[7] for instance in dataset])
pH = np.array([instance[8] for instance in dataset])
sulphates = np.array([instance[9] for instance in dataset])
alcohol = np.array([instance[10] for instance in dataset])
quality = np.array([instance[11] for instance in dataset])

high_quality_wine_index = [1 if quality[i] > 7 else 0 for i in range(len(quality))]
low_quality_wine_index = [1 if quality[i] < 4 else 0 for i in range(len(quality))]

print("Pearson correlation coefficients with quality")
print("Fixed Acidity: ", pearsonr(fixed_acidity, quality))
print("Volatile Acidity: ", pearsonr(volatile_acidity, quality))
print("Citric Acid: ", pearsonr(citric_acid, quality))
print("Residual Sugar: ", pearsonr(residual_sugar, quality))
print("Chlorides: ", pearsonr(chlorides, quality))
print("Free Sulfur Dioxide: ", pearsonr(free_sulfur_dioxide, quality))
print("Total Sulfur Dioxide: ", pearsonr(total_sulfur_dioxide, quality))
print("Density: ", pearsonr(density, quality))
print("pH: ", pearsonr(pH, quality))
print("Sulphates: ", pearsonr(sulphates, quality))
print("Alcohol: ", pearsonr(alcohol, quality))