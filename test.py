# importing the required libraries
from firebase import firebase
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Reading the csv file
dataset = pd.read_csv('cpdata.csv')


# Creating dummy variable for target i.e label
x = dataset.iloc[:, 0:-1].values  # features
y = dataset.iloc[:, -1].values  # labels


# Dividing the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Using standard scaler on features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using label encoder on labels
labelEncoder = LabelEncoder()
labelEncoder.fit(y_train)
y_train = labelEncoder.transform(y_train)
y_test = labelEncoder.transform(y_test)


# Importing Decision Tree classifier
clf = DecisionTreeClassifier()


# Fitting the classifier into training set
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


# Finding the accuracy of the model
a = accuracy_score(y_test, pred)
print("The accuracy of this model is: ", a * 100)


# Using firebase to import data to be tested
firebase = firebase.FirebaseApplication('https://crop-prediction-d2d98-default-rtdb.firebaseio.com/')
tp = firebase.get('/Realtime', None)

ah = tp['Air Humidity']
atemp = tp['Air Temp']
shum = tp['Soil Humidity']
pH = tp['Soil pH']
rain = tp['Rainfall']

predictcrop = [ah, atemp, pH, rain]
# converting list to numpy ndarray
predictcrop = np.array(predictcrop, dtype="float32")

# Predicting the crop
# using standard scaler to rescale the sample before predicting on it.
predictcrop = sc.transform(predictcrop.reshape(1, -1))
predictions = clf.predict(predictcrop)
# using labelEncoder's inverse_transform method to get class name
final_prediction = labelEncoder.inverse_transform(predictions)[0]
print('The predicted crop is ' + final_prediction)
# Sending the predicted crop to database
cp = firebase.put('/croppredicted', 'crop', final_prediction)
