import pandas as pd

import numpy as np
x = np.random.rand(185, 10)
y = np.random.rand(185)

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

pca.explained_variance_ratio_