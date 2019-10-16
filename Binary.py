import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 25
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

cr_batches = 10

label1 = 2
label2 = 3
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

F1_train, F1_test, T1_train, T1_test = train_test_split(F1, T1, test_size=0.20,random_state=100)

def conventionalMethod(X_train,X_test,y_train,y_test):
	classifier = svm.SVC(kernel = 'rbf',gamma=0.007,C=0.5)
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	y_pred1 = classifier.predict(X_train)
	accuracy_test = metrics.accuracy_score(y_test,y_pred)
	accuracy_train = metrics.accuracy_score(y_train,y_pred1)
	classifier.support_.sort()
	print(classifier.support_)
	print(classifier.support_.shape)
#	print(Y[classifier.support_])
	cm = metrics.confusion_matrix(y_test,y_pred)
	precision = metrics.precision_score(y_test,y_pred)
	recall = metrics.recall_score(y_test,y_pred)	
	return accuracy_test,accuracy_train,cm,precision,recall

def cross_validation(X,Y):
	accuracy_cr_test = 0
	accuracy_cr_train = 0
	precision_cr = 0
	recall_cr = 0
	cm_cr = np.zeros((2,2))
	cr_batch_size = int(X.shape[0]/cr_batches)
	for k in range(0,cr_batches):
		X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
		X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
		y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
		y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
		classifier = svm.SVC(kernel = 'linear',C=0.9,gamma='auto')
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy_cr_test = accuracy_cr_test + metrics.accuracy_score(y_test,y_pred)
		y_pred1 = classifier.predict(X_train)
		accuracy_cr_train = accuracy_cr_train + metrics.accuracy_score(y_train,y_pred1)
		precision_cr = precision_cr + metrics.precision_score(y_test,y_pred)
		recall_cr = recall_cr + metrics.recall_score(y_test,y_pred)
		cm_cr = cm_cr + metrics.confusion_matrix(y_test,y_pred)
	return accuracy_cr_test/cr_batches,accuracy_cr_train/cr_batches,cm_cr/cr_batches,precision_cr/cr_batches,recall_cr/cr_batches
	
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
			accuracy_mat1.append(conventionalMethod(F1,T1)[0])
#			print(i,j)
		tmp = max(accuracy_mat1)
		tmp_in = accuracy_mat1.index(tmp)
		if(tmp_in>=i):
			tmp_in = tmp_in+1
		print("For "+str(i)+" best accuracy is "+str(tmp)+" at 2nd label "+str(tmp_in))
		accuracy_mat.append(tmp)
	plt.plot(label_mat,accuracy_mat)
	plt.show

def print_params_conv(X,Y,Z,W):
	params = conventionalMethod(X,Y,Z,W)
	print("Conventional method:")
	print("Test Accuracy = "+str(params[0]))
	print("Train Accuracy = "+str(params[1]))
	print(params[2])
	print("Precision = "+str(params[3]))
	print("Recall = "+str(params[4]))

def print_params_cr(X,Y):
	params = cross_validation(X,Y)
	print("Cross validation method:")
	print("Test Accuracy = "+str(params[0]))
	print("Train Accuracy = "+str(params[1]))
	print(params[2])
	print("Precision = "+str(params[3]))
	print("Recall = "+str(params[4]))
	
def svc_param_selection(X, y, nfolds=5):
#	gammas = [0.000001, 0.00001]
#	degrees = [3, 4, 5, 6]
	param_grid = {'C':[0.001,0.01,0.03,0.05,0.1,0.5,0.8,1],'gamma':[0.00001,0.0001,0.001,0.005,0.007,0.008,0.01,0.1,1]}
	grid_search_classifier = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, scoring='accuracy', cv=nfolds,n_jobs=-1,refit = True)
	grid_search_classifier.fit(X, y)
	print(grid_search_classifier.best_params_)
	print(grid_search_classifier.best_score_)
	
def hyperparameter_variation(X,Y):
	Cs=[0.001,0.008,0.009,0.01,0.03,0.1,1,10]
	train_mat=[]
	test_mat=[]
	cr_batch_size = int(X.shape[0]/cr_batches)
	for i in Cs:
		accuracy_cr_test = 0
		accuracy_cr_train = 0
		for k in range(0,cr_batches):
			X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
			X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
			y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
			y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
			classifier = svm.SVC(kernel = 'linear',C=i,gamma='auto')
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			accuracy_cr_test = accuracy_cr_test + metrics.accuracy_score(y_test,y_pred)
			y_pred1 = classifier.predict(X_train)
			accuracy_cr_train = accuracy_cr_train + metrics.accuracy_score(y_train,y_pred1)
		train_mat.append(accuracy_cr_train/cr_batches)
		test_mat.append(accuracy_cr_test/cr_batches)
	plt.figure(figsize=(12,9))
	plt.plot(np.log10(Cs),train_mat,c='b')
	plt.plot(np.log10(Cs),test_mat,c='r')
	plt.show()
	
#print_params_conv(F1_train,F1_test,T1_train,T1_test)
#print_params_cr(F1,T1)
#best_label_pair()
#svc_param_selection(F1_train,T1_train)
hyperparameter_variation(F1,T1)