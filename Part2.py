import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 

Dataset = pd.read_csv('./train_set.csv',header=None);
Testset = pd.read_csv('./test_set.csv',header = None)
num_features = 25
F = Dataset.iloc[:,:num_features].values
s = StandardScaler()
F = s.fit_transform(F)
T = Dataset.iloc[:,-1].values
F1 = Testset.iloc[:,:num_features].values
F1 = s.fit_transform(F1)

cr_batches = 10
F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)

def conventionalMethod(X_train,X_test,y_train,y_test):
	classifier = svm.SVC(kernel = 'rbf',C=8.6,gamma=0.06)
	classifier.fit(X_train,y_train)
	y_pred = classifier.predict(X_test)
	accuracy_test = metrics.accuracy_score(y_test,y_pred)
	cm = metrics.confusion_matrix(y_test,y_pred)
	y_pred1 = classifier.predict(X_train)
	accuracy_train = metrics.accuracy_score(y_train,y_pred1)
#	precision = metrics.precision_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
#	recall = metrics.recall_score(y_test,y_pred,labels = range(0,10),average = 'weighted')
	return accuracy_test,accuracy_train,cm #precision,recall
	
def cross_validation(X,Y):
	accuracy_cr_test = 0
	accuracy_cr_train = 0
#	precision_cr = 0
#	recall_cr = 0
	cm_cr = np.zeros((10,10))
	cr_batch_size = int(X.shape[0]/cr_batches)
	for k in range(0,cr_batches):
		X_test = X[k*cr_batch_size:(k+1)*cr_batch_size,:]
		X_train = np.delete(X, np.s_[k*cr_batch_size:(k+1)*cr_batch_size],0)
		y_test = Y[k*cr_batch_size:(k+1)*cr_batch_size]
		y_train = np.delete(Y, np.s_[k*cr_batch_size:(k+1)*cr_batch_size])
		classifier = svm.SVC(kernel = 'rbf',C=8.6,gamma=0.06)
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy_cr_test = accuracy_cr_test + metrics.accuracy_score(y_test,y_pred)
		y_pred1 = classifier.predict(X_train)
		accuracy_cr_train = accuracy_cr_train + metrics.accuracy_score(y_train,y_pred1)
#		precision_cr = precision_cr + metrics.precision_score(y_test,y_pred)
#		recall_cr = recall_cr + metrics.recall_score(y_test,y_pred)
		cm_cr = cm_cr + metrics.confusion_matrix(y_test,y_pred)
	return accuracy_cr_test/cr_batches,accuracy_cr_train/cr_batches,np.round(cm_cr/cr_batches)#,precision_cr/cr_batches,recall_cr/cr_batches

def print_params_conv(X,Y,Z,W):
	params = conventionalMethod(X,Y,Z,W)
	print("Conventional method:")
	print("Test Accuracy = "+str(params[0]))
	print("Training Accuracy = "+str(params[1]))
	print(params[2])
#	print("Precision = "+str(params[2]))
#	print("Recall = "+str(params[3]))

def print_params_cr(X,Y):
	params = cross_validation(X,Y)
	print("Cross validation method:")
	print("Test Accuracy = "+str(params[0]))
	print("Training Accuracy = "+str(params[1]))
	print(params[2])
#	print("Precision = "+str(params[2]))
#	print("Recall = "+str(params[3]))

def write_test_ans_csv(X_train,Y_train,X_test):
	classifier = svm.SVC(kernel = 'rbf',C=8.6,gamma=0.06)
	classifier.fit(X_train,Y_train)
	y_pred = classifier.predict(X_test)
	print(len(y_pred))
	data = {'id':list(range(2000)), 'class':y_pred}
	sub = pd.DataFrame(data)
	sub.to_csv('./result.csv',index = None)
	
def svc_param_selection(X, y, nfolds=5):
	param_grid = {'gamma':[0.05,0.059,0.06,0.061]}
	grid_search_classifier = GridSearchCV(svm.SVC(kernel='rbf',C=8.6), param_grid, scoring='accuracy', cv=nfolds,n_jobs=-1,refit = True)
	grid_search_classifier.fit(X, y)
	print(grid_search_classifier.best_params_)
	print(grid_search_classifier.best_score_)

#print_params_conv(F_train,F_test,T_train,T_test)
#print_params_cr(F,T)
write_test_ans_csv(F,T,F1)
#svc_param_selection(F_train,T_train)