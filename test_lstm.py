import pandas as pd
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LstmModel
from utils import series_to_supervised, _calculate_mape

WINDOW_SIZE = 7 * 48
NUM_OF_FEATURES = 12
EPOCHS = 30



# load dataset
dataset = read_csv('./preprocessed/TrainActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')
submission = read_csv('./preprocessed/SampleSubmission3.csv', header=0, index_col='ConsumptionDate')
test = read_csv('./preprocessed/TestActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')
test['gen_sum'] = (test['Gen1num1'] + test['Gen1num2']) * 0.03 + test['Gen2'] * 0.06
test_values = test.drop(['Gen1num1', 'Gen1num2', 'Gen2'], axis=1)
dataset = dataset.drop(['Gen1num1', 'Gen1num2', 'Gen2'], axis=1)

# print(test_values.columns)
# print(dataset.columns)
test_values = pd.concat([pd.DataFrame(dataset).iloc[(-WINDOW_SIZE):, :], pd.DataFrame(test_values)], axis=0)

test_values = test_values.values

# dataset.reset_index()
# dataset.reindex(columns=['ConsumptionDate'])
values = dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(values)
test_scaled = scaler.transform(test_values)
train_scaled = pd.DataFrame(train_scaled)
test_scaled = pd.DataFrame(test_scaled)

# frame as supervised learning
train_X, train_y = series_to_supervised(train_scaled, WINDOW_SIZE, 1)
# drop columns we don't want to predict

# test as supervised learning
test_X, test_y = series_to_supervised(test_scaled, WINDOW_SIZE, 1)

model = LstmModel(train_X.shape[1], train_X.shape[2]).model
model.load_weights('best.h5')
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

test_X = test_X.reshape((test_X.shape[0], NUM_OF_FEATURES * WINDOW_SIZE))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0:NUM_OF_FEATURES-1]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 0:NUM_OF_FEATURES-1]), axis=1)
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

submission['ActualConsumption'] = inv_yhat

submission.to_csv('submission.csv')
