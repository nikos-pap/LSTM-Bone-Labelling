import numpy as np
from model import LSTMmodel


if __name__ == '__main__':
	inputs = np.load('training/inputs.npy')
	outputs = np.load('training/outputs.npy')
	model = LSTMmodel()
	model.build(inputs.shape[1], outputs.shape[1])
	model.train(inputs, outputs, 500)
	model.save()
	# model = LSTMmodel.load()
	inputs = np.load('evaluation/inputs.npy')
	outputs = np.load('evaluation/outputs.npy')
	print(inputs.shape, outputs.shape)
	model.evaluate(inputs, outputs)
	print(model.root_mean_square_error(inputs, outputs))
