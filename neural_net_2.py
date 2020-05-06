import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from numpy import sqrt
from numpy import mean
from numpy import std

def get_dataset(input_scaler, output_scaler):
    path = './neural_net_trial.csv'
    df = read_csv(path, header=0)

    # split into input and output columns
    X, y = df.values[:, :-1], df.values[:, -1]

    # split into train and test
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.33)
    print(trainX.shape, testX.shape, trainy.shape, testy.shape)

    if input_scaler is not None:
        input_scaler.fit(trainX)
        trainX = input_scaler.transform(trainX)
        testX = input_scaler.transform(testX)
    if output_scaler is not None:
        # reshape 1d arrays to 2d arrays
        trainy = trainy.reshape(len(trainy), 1)
        testy = testy.reshape(len(testy), 1)
        # fit scaler on training dataset
        output_scaler.fit(trainy)
        # transform training dataset
        trainy = output_scaler.transform(trainy)
        # transform test dataset
        testy = output_scaler.transform(testy)
    return trainX, trainy, testX, testy

def evaluate_model(trainX, trainy, testX, testy):
    # define model
    model = Sequential()
    model.add(Dense(50, activation='relu'))
    for i in range(4):
        model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    model.fit(trainX, trainy, epochs=150, batch_size=20, verbose=0)

    error = model.evaluate(testX, testy, verbose=0)
    return error

def repeated_evaluation(input_scaler, output_scaler, n_repeats=10):
    trainX, trainy, testX, testy = get_dataset(input_scaler, output_scaler)
    results = list()
    for _ in range(n_repeats):
        test_mse = evaluate_model(trainX, trainy, testX, testy)
        #print('>%.3f' % test_mse)
        results.append(test_mse)
    return results

results_unscaled_outputs = repeated_evaluation(StandardScaler(), None)
#results_unscaled_inputs = repeated_evaluation(None, StandardScaler())
#results_normalized_inputs = repeated_evaluation(MinMaxScaler(), StandardScaler())
#results_standardized_inputs = repeated_evaluation(StandardScaler(), StandardScaler())
print(mean(results_unscaled_outputs))
##results = [results_unscaled_inputs, results_normalized_inputs, results_standardized_inputs]
##labels = ['unscaled', 'normalized', 'standardized']
##pyplot.boxplot(results, labels=labels)
##pyplot.show()
