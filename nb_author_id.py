#!/usr/bin/python

""" 
authors and labels:
- Enrique has label 0
- Juan has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


def get_model_train(features_train, labels_train):
	clf = GaussianNB()
	clf.fit(features_train, labels_train)
	return clf

def predict(clf, features_test):
	return clf.predict(features_test)

def get_accuracy(clf, features_test, labels_test):
	return clf.score(features_test, labels_test)

if __name__ == "__main__":
	### features_train and features_test are the features for the training
	### and testing datasets, respectively
	### labels_train and labels_test are the corresponding item labels
	features_train, features_test, labels_train, labels_test = preprocess()

	t0 = time()
	clf = get_model_train(features_train, labels_train)
	print("training time: {} s".format(round(time()-t0, 3)))
	t0 = time()
	print("Prediction: {}".format(predict(clf, features_test)))
	print("Accuracy: {}".format(get_accuracy(clf, features_test, labels_test)))
	print("prediction time: {} s".format(round(time()-t0, 3)))

