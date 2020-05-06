import numpy as np
import pandas as pd 
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

# get training data 
X_training = pd.read_csv("data/std_train_x.csv",header=None)
y_training = pd.read_csv("data/train_y.csv",header=None)

# train
model = sm.OLS(y_training, X_training).fit()

# get validation data
X_validation = pd.read_csv("data/std_val_x.csv",header=None)
y_validation = pd.read_csv("data/val_y.csv",header=None)

# validate
predictions = model.predict(X_validation)
rmse_pred = rmse(y_validation[0].values.tolist(), predictions.values.tolist())
print("the RMSE is:", rmse_pred)

