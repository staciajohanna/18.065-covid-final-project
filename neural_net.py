from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import math

# Process training set
# Loading data
path = './train_data.csv'
init_train_data = read_csv(path, header=0) # 109 x 10 matrix (header row not included)

# Split into input and output columns
# For X, the first column (country's name) is removed
X_train, y_train = init_train_data.values[:, 1:], init_train_data.values[:, -1]
X_train = tf.convert_to_tensor(np.asarray(X_train), np.float32)
y_train = tf.convert_to_tensor(np.asarray(y_train), np.float32)

# Process validation set
# Loading data
path = './validation_data.csv'
init_valid_data = read_csv(path, header=0)

# Split into input and output columns
X_valid, y_valid = init_valid_data.values[:, 1:], init_valid_data.values[:, -1]
X_valid = tf.convert_to_tensor(np.asarray(X_valid), np.float32)
y_valid = tf.convert_to_tensor(np.asarray(y_valid), np.float32)

# Number of features without PCA
num_features = X_train.shape[1]
print(num_features)

# Define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(num_features,)))
model.add(Dense(1))

# Compile model
model.compile(optimizer='sgd', loss='mse')

# Fit model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# Evaluate model with validation data
error = model.evaluate(X_valid, y_valid, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, math.sqrt(error)))
