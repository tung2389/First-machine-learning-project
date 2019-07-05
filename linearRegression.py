import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

def readAndHandleData():
    global data, simplified_data, predict
    data = pd.read_csv("student-mat.csv", sep=";")
    simplified_data = data[{"G1","G2","G3","studytime","failures","absences"}]
    predict = "G3"

def getFeaturesAndLabel():
    global x,y
    x = np.array(simplified_data.drop([predict], 1))
    y = np.array(simplified_data[predict])

def trainTheModel():
    global x_train, x_test, y_train, y_test, linear
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

def printAccuracyRate():
    accuracy_rate = linear.score(x_test, y_test)
    print("Accuracy rate: ", accuracy_rate)

def printMoreInfo():
    print("Coefficients: ", linear.coef_)
    print("Intercept: ", linear.intercept_)

def printAllPredictionsAndActualValues():
    predictions = linear.predict(x_test)
    for x in range(len(predictions)):
        print("Predictions: ", predictions[x], "Inputs: ", x_test[x], "Actual outputs: ", y_test[x])


readAndHandleData()
getFeaturesAndLabel()
trainTheModel()
printAccuracyRate()
printMoreInfo()
printAllPredictionsAndActualValues()