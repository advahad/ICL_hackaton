from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


class LstmModel:
    def __init__(self, x_shape, y_shape):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(x_shape, y_shape), activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')

