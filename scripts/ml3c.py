# Importing Libraries 
#------------------------------------# 
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas
#------------------------------------#


# Preprocessing and Feature Selection
#------------------------#
filename = 'winequality.csv'
# reading the file using pandas 
data_read = pandas.read_csv(filename, delimiter = ';')
print 'shape of data is ' , '\n' , data_read.shape

# for getting summary of data
df=pandas.DataFrame(data_read)
df = df.describe()
print df

# Getting histograms of every feature
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
for i in features:
	data = data_read[str(i)]
	plt.hist(data, bins='auto')  
	plt.title(str(i) + "Histogram")
#	plt.show()
data = data_read['quality']  
plt.hist(data, bins='auto')  
plt.title("Quality Histogram")
#plt.show()

#Feature Selection
# seeing what feature is mostly correlated to output feature
# getting correlation
for i in features:
	data1 = data_read[str(i)]
	data2 = data_read['quality']
	cor = np.corrcoef(data1, data2)[0,1]  
	print 'Quality correlation with', str(i), '::',  cor

# Adding a column of ones 
m=data_read.shape[0]
data_read = np.column_stack((np.ones((m, 1)), data_read))
print 'new data shape', data_read.shape


# after getting all features, we add/remove features which are of less importance
data_to_use = np.array(data_read)
print 'shape of data after converted to numpy array ', '\n' , data_to_use.shape
#[0,2,3,4,6,9,10,11]
X = data_to_use[:,[0,1,2,3,4,6,9,10,11]]
print 'printing X', '\n',  X
print 'shape of X','\n', X.shape

y = data_to_use[:,12]
print 'printing y' , '\n',  y
print 'y.shape', '\n', y.shape

# dividing the data in training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Normalizing the data for use
# scaler object has saved means aand standard deviations
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
print X.mean(axis=0) #0

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#---------------------------------------------#


# Applying Neural Network to traing the model using cross validation
#---------------------------------------------#
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=30, solver = 'adam', activation='relu', learning_rate_init=0.005)

## cross validation score on training set
# just to check regularisation
print 'Cross validation score:'
scores = cross_val_score(mlp, X_train,y_train, cv=5)  # scoring='f1_macro';;;less..   
print scores.mean()

print 'Cross validation f1 macro score:'
scores = cross_val_score(mlp, X_train,y_train, cv=5, scoring='f1_macro')  # scoring='f1_macro';;;less..   
print scores.mean()

# Fitting the model
mlp = mlp.fit(X_train, y_train)
#---------------------------------------------#

#Calculating metrics for evaluating our model
#---------------------------------------------#
# accuracy score
predicted = mlp.predict(X_test)
# confidence intervals
const = 1.96
error = 1 - (accuracy_score(y_test, predicted))
interval = const * np.sqrt( (error * (1 - error)) / len(X_test))
print 'accuracy score:'
print accuracy_score(y_test, predicted), '+/-', interval

# other metric
#f1-score
print 'f1-score:'
print f1_score(y_test, predicted, average = 'macro')



# Plotting Results
#---------------------------------------------#












# -----------------
#Residual sugar maybe is the most unusual distribution cause for a better understanding of the data
# a log10 has been applied and appears a bimodal distribution. The rest of them seems to be normal distributions,
# some of them right skewed.

#LogisticRegression(C=1e5)



