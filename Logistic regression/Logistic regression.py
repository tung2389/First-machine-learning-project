import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
np.random.seed(42)

IMG_SHAPE = 80
DESIRED_DIM = 80*80*3
#projection_matrix = np.random.randn(IMG_SHAPE * IMG_SHAPE, DESIRED_DIM) # Put it outside the loop to improve performance

def preprocess_img(img):
    resized_img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
    # grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # flatten_img = grayscale_img.flatten()
    flatten_img = resized_img.flatten()
    rescaled_img = flatten_img / 255.0
    final_img = rescaled_img
    # final_img = np.dot(rescaled_img, projection_matrix)
    return final_img

def getFilesInfo():
    base_dir = os.path.join(os.getcwd() + '\data\cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    cat_dir = os.path.join(train_dir, 'cats')
    dog_dir = os.path.join(train_dir, 'dogs')
    listFilesCat = os.listdir(cat_dir)
    listFilesDog = os.listdir(dog_dir)
    return [cat_dir, dog_dir, listFilesCat, listFilesDog]

def split_datasets(X, Y):
    X_train = X[100:]
    Y_train = Y[100:]
    X_test = X[:100]
    Y_test = Y[:100]
    return [X_train, Y_train, X_test, Y_test]

cat_dir, dog_dir, listFilesCat, listFilesDog = getFilesInfo()
numFiles = len(listFilesCat) + len(listFilesDog)

def saveFeaturesLabels():
    random.shuffle(listFilesCat)
    random.shuffle(listFilesDog)
    X = np.zeros((numFiles, DESIRED_DIM))
    Y = np.zeros(numFiles)
    i = 0
    for cat in listFilesCat:
        img_path = os.path.join(cat_dir.replace("\\", "/"), cat)
        img = cv2.imread(img_path)
        processed_img = preprocess_img(img)
        X[i] = processed_img
        Y[i] = 0
        i = i+1
    for dog in listFilesDog:
        img_path = os.path.join(dog_dir.replace("\\", "/"), dog)
        img = cv2.imread(img_path)
        processed_img = preprocess_img(img)
        X[i] = processed_img
        Y[i] = 1
        i = i+1
    
    with open(os.getcwd() + '/Logistic regression/features.csv','w') as data:
        np.savetxt(data, X)
    with open(os.getcwd() + '/Logistic regression/labels.csv','w') as data:
        np.savetxt(data, Y)
    
def loadDataSets():
    X = np.zeros((numFiles, DESIRED_DIM))
    Y = np.zeros(numFiles)
    with open(os.getcwd() + '/Logistic regression/features.csv','r') as data:
        X = np.loadtxt(data).reshape((numFiles, DESIRED_DIM))
    with open(os.getcwd() + '/Logistic regression/labels.csv','r') as data:
        Y = np.loadtxt(data).reshape(numFiles)
    X_train, Y_train, X_test, Y_test = split_datasets(X,Y)
    return [X_train, Y_train, X_test, Y_test]

def main():
    X_train, Y_train, X_test, Y_test = loadDataSets()
    # print(X_train.shape)
    # print(Y_train.shape)
    model = LogisticRegression(C=1e5)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    print(accuracy)

def plotTest():
    img = cv2.imread("d:/First-machine-learning-project/data/cats_and_dogs_filtered/train/cats/cat.2.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib use RGB, Opencv use BGR
    img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
    plt.figure(figsize=(100,100))
    plt.imshow(img)
    plt.show()

main()
