
#Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.neural_network import MLPClassifier

get_ipython().magic('matplotlib inline')

#Importing Data

data = pd.read_csv('/Users/anprahlad/Desktop/MachineLearning/CreditCardFraud/creditcard.csv')
data.head()


#Plotting Data
class_count = pd.value_counts(data['Class'], sort = 'True').sort_index()
class_count.plot(kind = 'bar')

plt.title("Fraud Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


#Feature Scaling

scale = preprocessing.StandardScaler()
data['normalizedAmount'] = scale.fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1)
data.head()


#Assigning X and Y

X = data.loc[:, data.columns != 'Class']
Y = data.loc[:, data.columns == 'Class']


#Number of and Indices of Data Points in Minority Class (Class == 1)
fraud_count = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

#Indices of Majority Class
normal_indices = data[data.Class == 0].index

#Picking Random Indices from Majority Class Indices (n = # of Fraud Points)
random_normal_indices = np.random.choice(normal_indices,fraud_count,replace = False)
random_normal_indices = np.array(random_normal_indices)

#Combining the two to create Undersampled Dataset
undersampled_indices = np.concatenate([fraud_indices, random_normal_indices])
undersampled_data = data.loc[undersampled_indices, :]

#Assigning X and Y for Undersampled Data Set
X_under = undersampled_data.loc[:, undersampled_data.columns != 'Class']
Y_under = undersampled_data.loc[:, undersampled_data.columns == 'Class']




#Plotting New UnderSampled Data
class_count_undersample = pd.value_counts(undersampled_data['Class'], sort = 'True').sort_index()
class_count_undersample.plot(kind = 'bar')

plt.title("Fraud Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")




#Splitting Data into Train and Test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

X_train_under, X_test_under, Y_train_under, Y_test_under = train_test_split(X_under, Y_under, test_size = 0.3, random_state = 0)

print("")
print("Number transactions train dataset: ", len(X_train_under))
print("Number transactions test dataset: ", len(X_test_under))
print("Total number of transactions: ", len(X_train_under)+len(X_test_under))


#Creating Multi-Layered Perceptron (UnderSampled Data)

mlp = MLPClassifier(hidden_layer_sizes = (28,28,28), activation = 'relu', learning_rate_init = 0.01, solver = 'adam')

#Fitting Data
mlp.fit(X_train_under, Y_train_under)

#Prediction
prediction = mlp.predict(X_test_under)

#Confusion Matrix and Classification Report
print(confusion_matrix(Y_test_under, prediction))
print(classification_report(Y_test_under, prediction))


# In[34]:


#Creating Multi-Layered Perceptron (Skewed Data)

mlp2 = MLPClassifier(hidden_layer_sizes = (28,28,28), activation = 'relu', learning_rate_init = 0.01, solver = 'adam')

#Fitting Data
mlp2.fit(X_train, Y_train)

#Prediction
prediction2 = mlp2.predict(X_test)

#Confusion Matrix and Classification Report
print(confusion_matrix(Y_test, prediction2))
print(classification_report(Y_test, prediction2))


#Creating Multi-Layered Perceptron (Whole Data)

mlp3 = MLPClassifier(hidden_layer_sizes = (28,28,28), activation = 'relu', learning_rate_init = 0.01, solver = 'adam')

#Fitting Data
mlp3.fit(X_train_under, Y_train_under)

#Prediction
prediction3= mlp.predict(X_test)

#Confusion Matrix and Classification Report
print(confusion_matrix(Y_test, prediction3))
print(classification_report(Y_test, prediction3))

