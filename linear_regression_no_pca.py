import numpy as np
import pandas as pd 
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

all_data = pd.read_csv("data/std_train_x.csv",header=None)
df = all_data
target = pd.read_csv("data/train_y.csv",header=None)

#training 
X_training = df
y_training = target
model = sm.OLS(y_training, X_training).fit()

predictions = model.predict(X_training)
rmse_pred = rmse(y_training[0].values.tolist(), predictions.values.tolist())
print("the RMSE is:", rmse_pred)
