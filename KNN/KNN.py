import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import os

def createValidPath(filename):
     path = os.path.join(os.path.dirname(__file__), filename)
     return path

def readData():
     global data
     path = createValidPath("car.data")
     data = pd.read_csv(path)

def transformDataSet(): 
     global preprocess, buying, maint, doors, persons, lug_boot, safety, _class
     preprocess = preprocessing.LabelEncoder()
     buying = preprocess.fit_transform(list(data["buying"]))
     maint = preprocess.fit_transform(list(data["maint"]))
     doors = preprocess.fit_transform(list(data["doors"]))
     persons = preprocess.fit_transform(list(data["persons"]))
     lug_boot = preprocess.fit_transform(list(data["lug_boot"]))
     safety = preprocess.fit_transform(list(data["safety"]))
     _class = preprocess.fit_transform(list(data["class"]))

def createFeaturesAndLabels():
     global x, y
     x = list(zip(buying, maint, doors, persons, lug_boot, safety))
     y = list(_class)

def trainTheModel():
     global predict, x_train, x_test, y_train, y_test, model
     predict = "class"
     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
     model = KNeighborsClassifier(n_neighbors=9)
     model.fit(x_train, y_train)

def printMoreModelInfo():
     accuracy_rate = model.score(x_test, y_test)
     print("Accuracy rate: ", accuracy_rate)
     predicted = model.predict(x_test)
     names = ["unacc", "acc", "good", "vgood"]
     for i in range(len(predicted)):
          print("Features: ", x_test[i], "Predicted label: ", names[predicted[i]], "Actual label: ", names[y_test[i]])

readData()
transformDataSet()
createFeaturesAndLabels()
trainTheModel()
printMoreModelInfo()
