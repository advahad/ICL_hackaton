import pandas as pd
from keras.callbacks import *
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from utils import series_to_supervised

WINDOW_SIZE = 7 * 48
NUM_OF_FEATURES = 12
EPOCHS = 30

class LstmModel:
    def __init__(self, x_shape, y_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(x_shape, y_shape)))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')