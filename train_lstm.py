import pandas as pd
from keras.callbacks import *
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LstmModel
from utils import series_to_supervised

WINDOW_SIZE = 7 * 48
NUM_OF_FEATURES = 38
EPOCHS = 30




# load dataset
train_cons = read_csv('./preprocessed/TrainActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')
train_cons.index = pd.to_datetime(train_cons.index)
print(train_cons[train_cons['ActualConsumption'] < 60])
train_prod = read_csv('./preprocessed/TrainProdDataP.csv', header=0, index_col='ProductionDate')
planned = read_csv('./preprocessed/TrainPlannedDailyProduction.csv', header=0, index_col='ProductionDate')

train_prod.index = pd.to_datetime(train_prod.index)
train_columns = ['temp', 'barometric']
dataset = train_cons.merge(train_prod[train_columns], left_index=True, right_index=True, how='inner')
submission = read_csv('./preprocessed/SampleSubmission3.csv', header=0, index_col='ConsumptionDate')

test_cons = read_csv('./preprocessed/TestActualConsumptionDataP.csv', header=0, index_col='ConsumptionDate')
# test_prod = read_csv('./preprocessed/TestProdDataP.csv', header=0, index_col='ProductionDate')
# test = pd.concat([test_cons, test_prod], axis=0)
#
# test['gen_sum'] = (test['Gen1num1'] + test['Gen1num2']) * 0.03 + test['Gen2'] * 0.06
# test_values = test.drop(['Gen1num1', 'Gen1num2', 'Gen2'], axis=1)
dataset = dataset.drop(['Gen1num1', 'Gen1num2', 'Gen2'] + train_columns, axis=1)

# print(test_values.columns)
print(dataset.columns)
print(dataset.shape)
# test_values = pd.concat([pd.DataFrame(dataset).iloc[(-WINDOW_SIZE):, :], pd.DataFrame(test_values)], axis=0)
#
# test_values = test_values.values

# dataset.reset_index()
# dataset.reindex(columns=['ConsumptionDate'])
values = dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(values)
# test_scaled = scaler.transform(test_values)
train_scaled = pd.DataFrame(train_scaled)
# test_scaled = pd.DataFrame(test_scaled)

# frame as supervised learning
train_X, train_y = series_to_supervised(train_scaled, WINDOW_SIZE, 1)
# drop columns we don't want to predict

# fit network
cp = ModelCheckpoint('best.h5', save_best_only=True)
es = EarlyStopping(patience=5)
rlrop = ReduceLROnPlateau(patience=2)
model = LstmModel(train_X.shape[1], train_X.shape[2]).model
history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=72, validation_split=0.2, verbose=2,
                    shuffle=False, callbacks=[cp, es, rlrop])
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


