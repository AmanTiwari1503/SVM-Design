import numpy as np
import pandas as pd
from cvxopt import matrix as cvx_matrix
from cvxopt import solvers as cvx_solvers
from sklearn.model_selection import train_test_split
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 10
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

label1 = 0
label2 = 1
F1 = []
T1 = []
for k in range(0,len(T)):
	if(T[k]==label1):
		F1.append(F[k])
		T1.append(-1)
	elif(T[k]==label2):
		F1.append(F[k])
		T1.append(1)
F1 = np.asarray(F1)
T1 = np.asarray(T1)

def linear(x1,x2):
	return np.dot(x1,x2)

def polynomial(x1,x2,p):
	return np.power((1+np.dot(x1,x2)),p)

def gaussian(x1,x2,gamma):
	return np.exp(-1*gamma*(np.linalg.norm(x1-x2)**2))

class SVM(object):
	def __init__(self,kernel='linear',C=None,p=4,gamma=1):
		self.kernel = kernel
		self.C = C
		self.p = p
		self.gamma = gamma
		if self.C is not None:
			self.C = float(self.C)
	
	def fit(self,X,Y):
		n_samples = X.shape[0]
#		n_features = X.shape[1]
		
		K = np.zeros((n_samples,n_samples))
		if self.kernel is 'linear':
			for i in range(n_samples):
				for j in range(n_samples):
					K[i,j] = linear(X[i],X[j])
		
		elif self.kernel is 'poly':
			for i in range(n_samples):
				for j in range(n_samples):
					K[i,j] = polynomial(X[i],X[j],self.p)
		
		elif self.kernel is 'rbf':
			for i in range(n_samples):
				for j in range(n_samples):
					K[i,j] = gaussian(X[i],X[j],self.gamma)
		
		else:
			raise Exception('Invalid kernel')
		
		#Declaring parameters for CVX solvers
		P = cvx_matrix(np.outer(Y,Y)* K)
		q = cvx_matrix(-np.ones((n_samples,1)))
		Y = Y.astype('double')
		A = cvx_matrix(Y.reshape((1,-1)))
		b = cvx_matrix(0.0)
		
		if self.C is None:
			G = cvx_matrix(-np.identity(n_samples))
			h = cvx_matrix(np.zeros((n_samples,1)))	
		else:
			t1 = -np.identity(n_samples)
			t2 = np.identity(n_samples)
			G = cvx_matrix(np.vstack((t1,t2)))
			t1 = np.zeros((n_samples,1))
			t2 = np.ones((n_samples,1))*self.C
			h = cvx_matrix(np.vstack((t1,t2)))
		
		answer = cvx_solvers.qp(P,q,G,h,A,b)
		lag_mul = np.ravel(answer['x'])
		support_vectors = lag_mul>1e-5
		indexes = np.arange(len(lag_mul))
		lag_indexes = indexes[support_vectors]
		self.lag_mul = lag_mul[support_vectors]
		self.sv_x = X[support_vectors]
		self.support_vectors_=self.sv_x.astype()
#		for i in range(self.sv_x.shape[0]):
#			self.support_vectors_.append(float(self.sv_x[i]))
		print(self.support_vectors_)
		self.sv_y = Y[support_vectors]
#		print(self.sv_x,self.sv_y)
		print('No. of support vectors = '+str(len(self.lag_mul)))
		
		#intercept
		self.b = 0
		for i in range(len(self.lag_mul)):
#			print(K[lag_indexes[i],lag_indexes].shape)
			self.b = self.sv_y[i] - np.sum(self.lag_mul*self.sv_y*K[lag_indexes[i],lag_indexes])
#			print(self.b)
		self.b = self.b/len(self.lag_mul)
		
	def y_prediction(self,X):
		y_pred = np.zeros(X.shape[0])
		print(X.shape[0])
		if self.kernel is 'linear':
			for i in range(X.shape[0]):
				for j in range(len(self.lag_mul)):
					y_pred[i] += self.lag_mul[j]*self.sv_y[j]*linear(X[i],self.sv_x[j])
		
		elif self.kernel is 'poly':
			for i in range(X.shape[0]):
				for j in range(len(self.lag_mul)):
					y_pred[i] += self.lag_mul[j]*self.sv_y[j]*polynomial(X[i],self.sv_x[j],self.p)
		
		elif self.kernel is 'rbf':
			for i in range(X.shape[0]):
				for j in range(len(self.lag_mul)):
					y_pred[i] += self.lag_mul[j]*self.sv_y[j]*gaussian(X[i],self.sv_x[j],self.gamma)
		
		return y_pred+self.b
	
	def predict(self,X):
		y_p = self.y_prediction(X)
		y_ans = np.zeros(X.shape[0])
		for i in range(len(y_p)):
			if y_p[i] >= 0:
				y_ans[i] = 1
			elif y_p[i] < 0:
				y_ans[i] = -1
		return y_ans
		
		
		
if __name__ == "__main__":
	def conventionalMethod(X,Y):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=100)
		classifier = SVM(kernel='linear',C=0.1)
		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)
		accuracy = metrics.accuracy_score(y_test,y_pred)
		print('Accuracy is ='+str(accuracy))
	
	conventionalMethod(F1,T1)
		