from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


WINDOW_SIZE = 48
NUM_OF_FEATURES = 12
EPOCHS = 1

def _calculate_mape(Y_real, Y_pred):
    return np.sum(np.abs(Y_real - Y_pred)) / np.sum(Y_pred)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg




# load dataset
dataset = read_csv('./preprocessed/TrainActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')
submission = read_csv('./preprocessed/SampleSubmission3.csv', header=0, index_col='ConsumptionDate')
test = read_csv('./preprocessed/TestActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')

# dataset.reset_index()
# dataset.reindex(columns=['ConsumptionDate'])
dataset = dataset.drop(['Unnamed: 0', 'Gen1num1', 'Gen1num2', 'Gen2'], axis=1)
values = dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, WINDOW_SIZE, 1)
# drop columns we don't want to predict
reframed = reframed.iloc[:, :-11]


# print(reframed.head())

# split into train and test sets
values = reframed.values

# train = values[:7248, :]
# test = values[7248:, :]
# # split into input and outputs
# train_X, train_y = train[:, 1:], train[:, 0]
# test_X, test_y = test[:, 1:], test[:, 0]

# split into input and outputs
train_X, train_y = values[:, 1:], values[:, 0]
test_X, test_y = values[:, 1:], values[:, 0]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], NUM_OF_FEATURES, WINDOW_SIZE))
test_X = test_X.reshape((test_X.shape[0], NUM_OF_FEATURES, WINDOW_SIZE))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# fit network
history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=72, validation_split=0.2, verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)

# plot results
pyplot.plot(test_y, label='actual')
pyplot.plot(yhat, label='predicted')
pyplot.legend()
pyplot.show()



mape = _calculate_mape(test_y, yhat)
print(yhat)
print('Test MAPE: %.3f' % mape)

rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)


test_X = test_X.reshape((test_X.shape[0], NUM_OF_FEATURES*WINDOW_SIZE))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0:11]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 0:11]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('-------Inverse RMSE: %.3f' % rmse)

mape = _calculate_mape(inv_y, inv_yhat)
print(inv_yhat)
print('-------Inverse MAPE: %.3f' % mape)

# plot results
pyplot.plot(inv_y, label='Inverse actual')

pyplot.plot(inv_yhat, label='Inverse predicted')
pyplot.legend()
pyplot.show()

 # submition



