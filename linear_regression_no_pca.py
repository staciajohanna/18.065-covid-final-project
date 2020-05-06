import numpy as np
import pandas as pd 
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

# numpy csv-reading explorations: 
'''numpy_array = np.genfromtxt("train_data.csv", delimiter=";", dtype=None, skip_header=1)
print(numpy_array.dtype)'''

# pandas explorations
all_data = pd.read_csv("data/std_train_x.csv",header=None)
df = all_data
target = pd.read_csv("data/train_y.csv",header=None)
# df = all_data[["humidityNormalized", "tempCNormalized", "GDPNormalized", "PopulationDensityNormalized", "MedianAgeNormalized", "LandAreaNormalized", "GHSINormalized", "SARSDeathsNormalized"]]
# target = all_data[["PerCapitaCOVIDDeathsNormalized"]]

X_training = df
y_training = target
model = sm.OLS(y_training, X_training).fit()
predictions = model.predict(X_training)
print(model.summary())
rmse_pred = rmse(y_training[0].values.tolist(), predictions.values.tolist())
# print(y_training[0].values.tolist())
# rmse_pred = sm.tools.eval_measures.rmse(y_training.values.tolist(), predictions, axis=0)
print("the RMSE is:", rmse_pred)
