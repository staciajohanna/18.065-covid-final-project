from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_y = pd.read_csv("data//train_y.csv")
val_y = pd.read_csv("data//val_y.csv")
alpha = np.linspace(0.01,0.4,10)
alpha = np.linspace(0.01,0.4,10)

for id in ['std', 'pc2', 'pc4', 'pc6']:
	train_x = pd.read_csv(f'data//{id}_train_x.csv')
	val_x = pd.read_csv(f'data//{id}_val_x.csv')
	r2_train =[]
	r2_val =[]
	norm = []
	rmse_train = []
	rmse_val = []
	for i in range(10):
		lasso = Lasso(alpha = alpha[i])
		lasso.fit(train_x, train_y)
		train_predict = lasso.predict(train_x)
		val_predict = lasso.predict(val_x)

		r2_train = np.append(r2_train, r2_score(train_predict,train_y))
		r2_val = np.append(r2_val, r2_score(val_predict,val_y))
		norm = np.append(norm,np.linalg.norm(lasso.coef_))
		rmse_train.append(mean_squared_error(train_predict, train_y)**0.5)
		rmse_val.append(mean_squared_error(val_predict, val_y)**0.5)
	print ("\n", id)
	print (f"r2_train: {min(r2_train)} r2_val: {min(r2_val)} rmse_train: {min(rmse_train)} rmse_val: {min(rmse_val)}")
	print ("r2_train : ", r2_train)
	print ("r2_val : ", r2_val)
	print ("rmse_train : ", rmse_train)
	print ("rmse_val : ", rmse_val)