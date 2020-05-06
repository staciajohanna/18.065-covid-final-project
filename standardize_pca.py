# Randomize/ shuffle
# Split into train, test, validation
# Standradize train, test, validation
# PCA train, test, validation
import pandas as pd
import numpy as np
# Read the CSV file 
all_data = pd.read_csv('all_data.csv')
x = all_data.filter(items = ['humidity', 'temp', 'ghsi', 'medianage', 'sars', 'landarea', 'gdp', 'popdens'])
x['popdens'] = pd.to_numeric(x['popdens'],errors='coerce')
y = all_data.filter(items = ['covidpercapita'])
x = x.values
y = y.values

# print ("x:", x)
# print ("y:", y)
# test_size: what proportion of original data is used for test set
# train_img, test_img, train_lbl, test_lbl = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.model_selection import train_test_split
train_val_x, test_x, train_val_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.25, random_state=0)

# We do not standardize the targets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()# Fit on training set only.
scaler.fit(train_x)# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(val_x)

# Standardized training data
np.savetxt(f'data//std_train_x.csv', train_x, delimiter = ',')
# Training targets
np.savetxt(f'data//train_y.csv', train_y, delimiter = ',')
# Normalized val data
np.savetxt(f'data//std_val_x.csv', val_x, delimiter = ',')
# Val targets
np.savetxt(f'data//val_y.csv', val_y, delimiter = ',')
# Normalized test data
np.savetxt(f'data//std_test_x.csv', test_x, delimiter = ',')
# Test targets
np.savetxt(f'data//test_y.csv', test_y, delimiter = ',')

from sklearn.decomposition import PCA# Make an instance of the Model
for n in [2, 4, 6]:
	pca = PCA(n)
	pca.fit(train_x)
	pca_train_x = pca.transform(train_x)
	pca_val_x = pca.transform(val_x)
	pca_test_x = pca.transform(test_x)
	explained_variance = pca.explained_variance_ratio_


	filename = f'data//pc{n}_ev{explained_variance}_train_x.csv'
	np.savetxt(filename, pca_train_x, delimiter = ',')

	filename = f'data//pc{n}_ev{explained_variance}_val_x.csv'
	np.savetxt(filename, pca_val_x, delimiter = ',')

	filename = f'data//pc{n}_ev{explained_variance}_test_x.csv'
	np.savetxt(filename, pca_test_x, delimiter = ',')