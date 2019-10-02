import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 10
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

cr_batches = 10 

def conventionalMethod(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=109)
	classifier = svm.SVC(kernel = 'linear')
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
#	precision = metrics.precision_score(y_test,y_pred)
#	recall = metrics.recall_score(y_test,y_pred)	
#	print("Conventional method:")
#	print("Accuracy = "+str(accuracy))
#	print("Precision = "+str(precision))
#	print("Recall = "+str(recall))
#	print("\n")
	return accuracy

def cross_validation(X,Y):
	accuracy_cr = 0
	precision_cr = 0
	recall_cr = 0
	cr_batch_size = int(X.shape[0]/cr_batches)
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
	
#conventionalMethod(F1,T1)
#cross_validation(F1,T1)
def best_label_pair():
	accuracy_mat = []
	label_mat = range(0,10)
	for i in range(0,10):
		accuracy_mat1 = []
		for j in range(0,10):
			if(i==j):
				continue
			label1 = i
			label2 = j
			F1 = []
			T1 = []
			for k in range(0,len(T)):
				if(T[k]==label1):
					F1.append(F[k])
					T1.append(0)
				elif(T[k]==label2):
					F1.append(F[k])
					T1.append(1)
			F1 = np.asarray(F1)
			T1 = np.asarray(T1)	
			accuracy_mat1.append(conventionalMethod(F1,T1))
#			print(i,j)
		tmp = min(accuracy_mat1)
		tmp_in = accuracy_mat1.index(tmp)
		if(tmp_in>=i):
			tmp_in = tmp_in+1
		print("For "+str(i)+" best accuracy is "+str(tmp)+" at 2nd label "+str(tmp_in))
		accuracy_mat.append(tmp)
	plt.plot(label_mat,accuracy_mat)
	plt.show
	
best_label_pair()