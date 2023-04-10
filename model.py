from keras.models import Model, load_model
from keras.layers import Layer, LSTM, Input, Dense, TimeDistributed, Reshape
from math import sqrt
import numpy as np


class LSTMmodel:
	def __init__(self):
		self.model = None

	def build(self, bone_num, points_num):
		inputs = Input(shape=(bone_num, 12))
		layer = LSTM(500, time_major=True, return_sequences=True)(inputs)
		layer = LSTM(2000, time_major=True, return_sequences=True)(layer)
		layer = LSTM(2000, time_major=True, return_sequences=True)(layer)
		layer = LSTM(points_num, time_major=True, return_sequences=True)(layer)
		layer = Reshape((points_num, bone_num))(layer)
		layer = TimeDistributed(Dense(3, activation='linear'))(layer)

		self.model = Model(inputs, layer)
		self.model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
		self.model.summary()

	def train(self, inputs, outputs, epochs=10):
		if self.model is None:
			print('build first')
			return
		print(inputs.shape, outputs.shape)
		self.model.fit(inputs, outputs, epochs=epochs, verbose=2)

	def save(self, path='./network'):
		self.model.save(path)

	@staticmethod
	def load(path='./network'):
		model = LSTMmodel()
		model.model = load_model(path)
		return model

	def evaluate(self, inputs, outputs):
		self.model.evaluate(inputs, outputs)

	def root_mean_square_error(self, inputs, outputs):
		approximations = self.model.predict(inputs)
		print(approximations.shape)
		fr = outputs - approximations
		fr0 = sqrt(np.sum(fr * fr))
		fr1 = sqrt(outputs.shape[0] * outputs.shape[1] * outputs.shape[2])
		return 100 * fr0 / fr1

	def distortion_percentage(self, inputs, outputs):
		approximations = self.model.predict(inputs)
		print(approximations.shape)
		fr = outputs - approximations
		fr0 = sqrt(np.sum(fr * fr))
		fr1 = np.linalg.norm(outputs - approximations, 'fro')
		return 100 * fr0 / fr1



def main():
	t = LSTMmodel(80, 5000)


if __name__ == '__main__':
	main()