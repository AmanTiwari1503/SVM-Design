import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 25
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

def conventionalMethod(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=109)
	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	cm = metrics.confusion_matrix(y_test,y_pred)
	print("Accuracy "+str(accuracy))
	print(cm)
	
conventionalMethod(F,T)