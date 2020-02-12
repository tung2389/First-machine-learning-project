import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import cv2
import os
np.random.seed(42)

IMG_SHAPE = 150
DESIRED_DIM = 500
projection_matrix = np.random.randn(IMG_SHAPE * IMG_SHAPE, DESIRED_DIM) # Put it outside the loop to improve performance

def preprocess_img(img):
    resized_img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    flatten_img = grayscale_img.flatten()
    rescaled_img = flatten_img / 255.0
    final_img = np.dot(rescaled_img, projection_matrix)
    return final_img

def prepareData():
    base_dir = os.path.join(os.getcwd() + '\data\cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    cat_dir = os.path.join(train_dir, 'cats')
    dog_dir = os.path.join(train_dir, 'dogs')
    listFilesCat = os.listdir(cat_dir)
    listFilesDog = os.listdir(dog_dir)
    X_train = np.zeros((len(listFilesCat) + len(listFilesDog), DESIRED_DIM))
    i = 0
    for cat in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir.replace("\\", "/"), cat)
        img = cv2.imread(img_path)
        processed_img = preprocess_img(img)
        X_train[i] = processed_img
        i = i+1
    for dog in os.listdir(dog_dir):
        img_path = os.path.join(dog_dir.replace("\\", "/"), dog)
        img = cv2.imread(img_path)
        processed_img = preprocess_img(img)
        X_train[i] = processed_img
        i = i+1
    
    
prepareData()
        

