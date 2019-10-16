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

F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=109)
def conventionalMethod(X_train,X_test,y_train,y_test):
	classifier = svm.SVC(kernel = 'linear',C=0.05)
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	y_pred1 = classifier.predict(X_train)
	accuracy_test = metrics.accuracy_score(y_test,y_pred)
	accuracy_train = metrics.accuracy_score(y_train,y_pred1)
	cm = metrics.confusion_matrix(y_test,y_pred)
#	precision = metrics.precision_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
#	recall = metrics.recall_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
	return accuracy_test,accuracy_train,cm
	
def cross_validation(X,Y):
	accuracy_cr_train = 0
	accuracy_cr_test = 0
#	precision_cr = 0
#	recall_cr = 0
	cm_cr = np.zeros((10,10))
	cr_batch_size = int(X.shape[0]/cr_batches)
	for k in range(0,cr_batches):
		X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
		X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
		y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
		y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
		classifier = svm.SVC(kernel = 'rbf', C = 10,gamma=0.04)
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy_cr_test = accuracy_cr_test + metrics.accuracy_score(y_test,y_pred)
		y_pred1 = classifier.predict(X_train)
		accuracy_cr_train = accuracy_cr_train + metrics.accuracy_score(y_train,y_pred1)
#		precision_cr = precision_cr + metrics.precision_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
#		recall_cr = recall_cr + metrics.recall_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
		cm_cr = cm_cr + metrics.confusion_matrix(y_test,y_pred)
	return accuracy_cr_test/cr_batches,accuracy_cr_train/cr_batches,np.round(cm_cr/cr_batches)#,precision_cr/cr_batches,recall_cr/cr_batches

def print_params_conv(X,Y,Z,W):
	params = conventionalMethod(X,Y,Z,W)
	print("Conventional method:")
	print("Test Accuracy = "+str(params[0]))
	print("Train Accuracy = "+str(params[1]))
	print(params[2])
#	print("Precision = "+str(params[3]))
#	print("Recall = "+str(params[4]))

def print_params_cr(X,Y):
	params = cross_validation(X,Y)
	print("Cross validation method:")
	print("Test Accuracy = "+str(params[0]))
	print("Train Accuracy = "+str(params[1]))
	print(params[2])
#	print("Precision = "+str(params[3]))
#	print("Recall = "+str(params[4]))

def svc_param_selection(X, y, nfolds=5):
#	gammas = [0.000001, 0.00001]
#	degrees = [3, 4, 5, 6]
	param_grid = {'C':[0.01,0.05,0.1,0.5,1,5,8,9,10,11,15],'gamma':[0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1]}
	grid_search_classifier = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, scoring='accuracy', cv=nfolds,n_jobs=-1,refit = True)
	grid_search_classifier.fit(X, y)
	print(grid_search_classifier.best_params_)
	print(grid_search_classifier.best_score_)

def hyperparameter_variation(X,Y):
	Cs=[0.1,0.5,1,5,10,50,100,1000]
#	Cs=[0.5,1,2,3,5,10,100]
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
			classifier = svm.SVC(kernel = 'rbf',C=i,gamma=0.04)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			accuracy_cr_test = accuracy_cr_test + metrics.accuracy_score(y_test,y_pred)
			y_pred1 = classifier.predict(X_train)
			accuracy_cr_train = accuracy_cr_train + metrics.accuracy_score(y_train,y_pred1)
		train_mat.append(accuracy_cr_train/cr_batches)
		test_mat.append(accuracy_cr_test/cr_batches)
	plt.figure(figsize=(12,9))
	plt.plot(np.log10(Cs),train_mat,c='b',label = 'Training accuracy')
	plt.plot(np.log10(Cs),test_mat,c='r',label = 'Test accuracy')
	plt.legend(loc='best')
	plt.title('Variation of C for Rbf kernel for '+str(num_features)+' features')
	plt.xlabel('log10(C)')
	plt.ylabel('Accuracy')
#	plt.savefig('./Pics/C_Variation_Rbf'+str(num_features)+'_'+str(label1)+'_'+str(label2)+'.png')
	plt.show()

#print_params_conv(F_train,F_test,T_train,T_test)
print_params_cr(F,T)
#svc_param_selection(F_train,T_train)
#hyperparameter_variation(F,T)