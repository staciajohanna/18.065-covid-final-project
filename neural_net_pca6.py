from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
import numpy as np
import tensorflow as tf
import math
#from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statistics

np.random.seed(42)

# Process training set
train_X_path = './data/pc6_ev[ 0.3413183 0.25478867 0.13134266 0.10615022 0.06533687 0.05613034]_train_x.csv'
X_train_raw = read_csv(train_X_path, header=None)
X_train = tf.convert_to_tensor(np.asarray(X_train_raw), np.float64)

train_y_path = './data/train_y.csv'
y_train_raw = np.asarray(read_csv(train_y_path, header=None))
y_train = tf.convert_to_tensor(y_train_raw, np.float64)

# Process validation set
valid_X_path = './data/pc6_ev[ 0.3413183 0.25478867 0.13134266 0.10615022 0.06533687 0.05613034]_val_x.csv'
X_valid_raw = read_csv(valid_X_path, header=None)
X_valid = tf.convert_to_tensor(np.asarray(X_valid_raw), np.float32)

valid_y_path = './data/val_y.csv'
y_valid_raw = np.asarray(read_csv(valid_y_path, header=None))
y_valid = tf.convert_to_tensor(y_valid_raw, np.float32)

num_features = X_train.shape[1]
num_loop = 5
total = 0.0

for i in range(num_loop):
    # Define model
    model = Sequential()
    model.add(Dense(40, activation='sigmoid', kernel_initializer='he_normal', input_shape=(num_features,)))
    #for i in range(2):
    #    model.add(Dense(10, activation='sigmoid', kernel_initializer='he_normal'))
    for i in range(4):
        model.add(Dense(40, activation='sigmoid', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))

    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adam", loss='mse')

    # Fit model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=10, verbose=0)

    # Evaluate model with validation data
    error = model.evaluate(X_valid, y_valid, verbose=0)
    total += math.sqrt(error)

print(total/num_loop)

#pyplot.title('Loss / Mean Squared Error')
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='validation')
#pyplot.legend()
#pyplot.show()
