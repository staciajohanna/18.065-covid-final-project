import numpy as np
import pandas as pd 
import statsmodels.api as sm

# numpy csv-reading explorations: 
'''numpy_array = np.genfromtxt("train_data.csv", delimiter=";", dtype=None, skip_header=1)
print(numpy_array.dtype)'''

# pandas explorations
all_data = pd.read_csv("train_data.csv")
df = all_data[["humidityNormalized", "tempCNormalized", "GDPNormalized", "PopulationDensityNormalized", "MedianAgeNormalized", "LandAreaNormalized", "GHSINormalized", "SARSDeathsNormalized"]]
target = all_data[["PerCapitaCOVIDDeathsNormalized"]]

X = df[["humidityNormalized", "tempCNormalized", "GDPNormalized", "PopulationDensityNormalized", "MedianAgeNormalized", "LandAreaNormalized", "GHSINormalized", "SARSDeathsNormalized"]]
y = target["PerCapitaCOVIDDeathsNormalized"]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())
