import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

training = pd.read_csv('training.csv')
x = training.filter(items = ['humidity', 'temp', 'ghsi', 'medianage', 'sars', 'popdens', 'landarea', 'gdp'])

for n in [2, 4, 6]:
	pca = PCA(n_components=n)
	principalComponents = pca.fit_transform(x)
	print(pca.explained_variance_ratio_)
	filename = f'pc{n}.csv'
	np.savetxt(filename, principalComponents, delimiter = ',')
