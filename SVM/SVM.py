import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def loadDataSet():
    global cancer
    cancer = datasets.load_breast_cancer()

def getFeaturesAndLabel():
    global x,y
    x = cancer.data
    y = cancer.target

def trainTheModel():
    global x_train, x_test, y_train, y_test,clf
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
    clf = svm.SVC(kernel="linear", C=2)
    clf.fit(x_train, y_train)

def printMoreInfo():
    classes = ["malignant", "benign"]
    y_prediction = clf.predict(x_test)
    accuracy_rate = metrics.accuracy_score(y_test, y_prediction)
    print("Accurary rate: ", accuracy_rate)
    for i in range(len(y_prediction)):
        print("Actual value: ", classes[y_test[i]], "   Predictions: ", classes[y_prediction[i]])

loadDataSet()
getFeaturesAndLabel()
trainTheModel()
printMoreInfo()