from sys import path as envpath
envpath.append('../utils/')
from data_manager import argmax
from constants import FRAME_NUM
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model
from keras.backend import dot, softmax, tanh, sum as ksum
from keras.layers import Layer, Dense, LSTM, Input, Dropout


class BoneLSTM:
	def __init__(self, custom_model=None):
		self.metrics = dict()
		if custom_model is None:
			custom_model = [['lstm', '1000'], ['dropout', '0.3'], ['lstm', '1000'], ['dropout', '0.2'], ['attention'], ['dense', '1300', 'relu'], ['dense', '700', 'relu'], ['dense', '300', 'relu'], ['dense', '100', 'relu'], ['dense', '5', 'softmax']]
		self.model = None
		self.create_from_string(custom_model)

	def create_from_string(self, layers):
		inputs = Input(shape=(FRAME_NUM, 3))
		layer = inputs
		for layer_data in layers:
			if layer_data[0] == 'lstm':
				layer = LSTM(int(layer_data[1]), return_sequences=True)(layer)
			if layer_data[0] == 'dropout':
				layer = Dropout(float(layer_data[1]))(layer)
			if layer_data[0] == 'attention':
				layer = AttentionLayer()(layer)
			if layer_data[0] == 'dense':
				layer = Dense(int(layer_data[1]), activation=layer_data[2])(layer)

		self.model = Model(inputs, layer)
		self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
		self.model.summary()

	def train(self, inputs, outputs, epochs: int = 100):
		callback = EarlyStopping(monitor='loss', patience=3)
		self.metrics = self.model.fit(inputs, outputs, epochs=epochs, batch_size=64, validation_split=0.2, verbose=2, callbacks=[callback])

	def predict(self, inputs):
		return self.model.predict(inputs)

	def evaluate(self, inputs, outputs):
		self.model.evaluate(inputs, outputs, batch_size=60)

	def get_groups(self, bones):
		groups: list[list[int]] = [[] for _ in range(5)]
		results = self.predict(bones)
		for index, outputs in enumerate(results):
			groups[argmax(outputs)].append(index)
		groups.insert(4, [])
		groups.insert(3, [])
		return groups

	def save(self, path='../network'):
		self.model.save(path)

	def load(self, path='../network'):
		if os.listdir(path):
			self.model = load_model(path)
			return
		print('Network Folder Empty. Please train the model first to save a network.')


class AttentionLayer(Layer):
	def __init__(self):
		super(AttentionLayer, self).__init__()

	def build(self, input_shape):
		self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
		self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)

	def call(self, inputs):
		u = tanh(dot(inputs, self.W) + self.b)
		alphas = softmax(u)
		result = ksum(inputs * alphas, axis=1)
		return result
