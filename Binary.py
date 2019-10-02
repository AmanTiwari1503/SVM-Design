import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import math

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 10
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

label1 = 6
label2 = 2
F1 = []
F2 = []

for i in range(0,len(T)):
	if(T[i]==label1):
		F1.append(F[i])
	elif(T[i]==label2):
		F2.append(F[i])

F1 = np.asarray(F1)
F2 = np.asarray(F2)