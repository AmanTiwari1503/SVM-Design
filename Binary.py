import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 10
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

label1 = 1
label2 = 3
F1 = []
T1 = []

for i in range(0,len(T)):
	if(T[i]==label1):
		F1.append(F[i])
		T1.append(0)
	elif(T[i]==label2):
		F1.append(F[i])
		T1.append(1)

F1 = np.asarray(F1)
T1 = np.asarray(T1)	

def conventionalMethod():
	X_train, X_test, y_train, y_test = train_test_split(F1, T1, test_size=0.30,random_state=109)
	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	precision = metrics.precision_score(y_test,y_pred)
	recall = metrics.recall_score(y_test,y_pred)	
	print("Accuracy = "+str(accuracy))
	print("Precision = "+str(precision))
	print("Recall = "+str(recall))
	
conventionalMethod()

