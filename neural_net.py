from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import math
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statistics

np.random.seed(42)

# Process training set
train_X_path = './data/std_train_x.csv'
X_train_raw = read_csv(train_X_path, header=None)
X_train = tf.convert_to_tensor(np.asarray(X_train_raw), np.float64)

train_y_path = './data/train_y.csv'
y_train_raw = np.asarray(read_csv(train_y_path, header=None))
y_train_transform = y_train_raw.transpose()[0]
y_train_mean = statistics.mean(y_train_transform)
y_train_stdev = statistics.stdev(y_train_transform)
y_train_raw = [((y-y_train_mean)/y_train_stdev) for y in y_train_raw]
y_train = tf.convert_to_tensor(y_train_raw, np.float64)

# Process validation set
valid_X_path = './data/std_val_x.csv'
X_valid_raw = read_csv(valid_X_path, header=None)
X_valid = tf.convert_to_tensor(np.asarray(X_valid_raw), np.float32)

valid_y_path = './data/val_y.csv'
y_valid_raw = np.asarray(read_csv(valid_y_path, header=None))
y_valid_transform = y_valid_raw.transpose()[0]
y_valid_mean = statistics.mean(y_valid_transform)
y_valid_stdev = statistics.stdev(y_valid_transform)
y_valid_raw = [((y-y_train_mean)/y_train_stdev) for y in y_valid_raw]
y_valid = tf.convert_to_tensor(y_valid_raw, np.float32)

# Number of features without PCA
num_features = X_train.shape[1]

# Define model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(num_features,)))
for i in range(10):
    model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse', metrics=["mse", "mae"])

# Fit model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=20, verbose=0)

# Evaluate model with validation data
error = model.evaluate(X_valid, y_valid, verbose=0)
print(error)
print(math.sqrt(error[0]))

#pyplot.title('Loss / Mean Squared Error')
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='validation')
#pyplot.legend()
#pyplot.show()
