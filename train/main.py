from sys import argv, path as envpath
from time import time
from matplotlib import pyplot as plt
envpath.append('..')
from utils.bone_model import BoneLSTM
from utils.data_manager import get_plot_name, load_data


def main():
	lstm = BoneLSTM()
	start = time()
	if len(argv) > 1 and argv[1] == '1':
		lstm.load('../network')
	else:
		print('Training Phase')
		training_data = load_data('../models/training2/')

		print(f'Training Shapes: {training_data[0].shape}, {training_data[1].shape}')
		lstm.train(training_data[0], training_data[1], epochs=200)

		plt.plot(lstm.metrics.history['accuracy'])
		plt.plot(lstm.metrics.history['loss'])
		plt.savefig(get_plot_name(path='../plots/'))
	
	if len(argv) > 1 and argv[1] == '2':
		lstm.save('../network')

	print('Evaluation Phase')
	evaluation_inputs, evaluation_outputs = load_data('../models/test2/', shuffle_lines=False)
	print(f'Shapes: {evaluation_inputs.shape}, {evaluation_outputs.shape}')
	lstm.evaluate(evaluation_inputs, evaluation_outputs)
	
	end = time()
	print(f'Total time: {end-start} seconds')


if __name__ == '__main__':
	main()
