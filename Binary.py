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
cr_batches = 10 

for i in range(0,len(T)):
	if(T[i]==label1):
		F1.append(F[i])
		T1.append(0)
	elif(T[i]==label2):
		F1.append(F[i])
		T1.append(1)

F1 = np.asarray(F1)
T1 = np.asarray(T1)	

def conventionalMethod(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=109)
	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	precision = metrics.precision_score(y_test,y_pred)
	recall = metrics.recall_score(y_test,y_pred)	
	print("Conventional method:")
	print("Accuracy = "+str(accuracy))
	print("Precision = "+str(precision))
	print("Recall = "+str(recall))
	print("\n")

def cross_validation(X,Y):
	accuracy_cr = 0
	precision_cr = 0
	recall_cr = 0
	cr_batch_size = int(F1.shape[0]/cr_batches)
	for k in range(0,cr_batches):
		X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
		X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
		y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
		y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
		classifier = svm.SVC(kernel = 'linear')
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy_cr = accuracy_cr + metrics.accuracy_score(y_test,y_pred)
		precision_cr = precision_cr + metrics.precision_score(y_test,y_pred)
		recall_cr = recall_cr + metrics.recall_score(y_test,y_pred)
	print("Cross validation method:")
	print("Accuracy = "+str(accuracy_cr/cr_batches))
	print("Precision = "+str(precision_cr/cr_batches))
	print("Recall = "+str(recall_cr/cr_batches))
	print("\n")
	
conventionalMethod(F1,T1)
cross_validation(F1,T1)
