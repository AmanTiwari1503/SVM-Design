import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

Dataset = pd.read_csv('./train_set.csv',header=None);
Testset = pd.read_csv('./test_set.csv',header = None)
num_features = 25
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values
F1 = Testset.iloc[:,:num_features].values

cr_batches = 10

def conventionalMethod(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10,random_state=109)
	classifier = svm.SVC(kernel = 'poly',C=0.05,degree=3)
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	cm = metrics.confusion_matrix(y_test,y_pred)
#	precision = metrics.precision_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
#	recall = metrics.recall_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
	return accuracy,cm #precision,recall
	
def cross_validation(X,Y):
	accuracy_cr = 0
#	precision_cr = 0
#	recall_cr = 0
	cm_cr = np.zeros((10,10))
	cr_batch_size = int(X.shape[0]/cr_batches)
	for k in range(0,cr_batches):
		X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
		X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
		y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
		y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
		classifier = svm.SVC(kernel = 'poly',C=0.05,degree=3)
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy_cr = accuracy_cr + metrics.accuracy_score(y_test,y_pred)
#		precision_cr = precision_cr + metrics.precision_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
#		recall_cr = recall_cr + metrics.recall_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
		cm_cr = cm_cr + metrics.confusion_matrix(y_test,y_pred)
	return accuracy_cr/cr_batches,np.round(cm_cr/cr_batches)#,precision_cr/cr_batches,recall_cr/cr_batches

def print_params_conv(X,Y):
	params = conventionalMethod(X,Y)
	print("Conventional method:")
	print("Accuracy = "+str(params[0]))
	print(params[1])
#	print("Precision = "+str(params[2]))
#	print("Recall = "+str(params[3]))

def print_params_cr(X,Y):
	params = cross_validation(X,Y)
	print("Cross validation method:")
	print("Accuracy = "+str(params[0]))
	print(params[1])
#	print("Precision = "+str(params[2]))
#	print("Recall = "+str(params[3]))

def write_test_ans_csv(X_train,Y_train,X_test):
	classifier = svm.SVC(kernel = 'poly',C=0.05,gamma='auto',degree = 3)
	classifier.fit(X_train,Y_train)
	y_pred = classifier.predict(X_test)
	print(len(y_pred))
	data = {'id':list(range(2000)), 'class':y_pred}
	sub = pd.DataFrame(data)
	sub.to_csv('./result.csv',index = None)
	
def svc_param_selection(X, y, nfolds):
	Cs = [0.1, 1, 5]
#	gammas = [0.000001, 0.00001]
	degrees = [3, 4, 5, 6]
	param_grid = {'C': Cs, 'degree': degrees}
	grid_search_classifier = GridSearchCV(svm.SVC(kernel='poly',gamma='auto'), param_grid, cv=nfolds)
	grid_search_classifier.fit(X, y)
	print(grid_search_classifier.best_params_)
	
#print_params_conv(F,T)
#print_params_cr(F,T)
write_test_ans_csv(F,T,F1)
#svc_param_selection(F,T,10)