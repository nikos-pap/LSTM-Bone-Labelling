from os import listdir, makedirs
from os.path import exists, isfile
from numpy.random import permutation
from numpy import concatenate, load, tile, array


def argmax(array):
	maxi = array[0]
	maxindex = 0
	for index, value in enumerate(array):
		if value > maxi:
			maxi = value
			maxindex = index
	return maxindex


def shuffle_data(*args):
	p = permutation(args[0].shape[0])
	return [data[p] for data in args]


def load_data(root_folder: str, shuffle_lines: bool = True):
	if not exists(root_folder):
		raise FileNotFoundError(f'Model folder "{root_folder}" not found.')
	
	inputs = []
	outputs = []

	for folder in listdir(root_folder):
		print(f'Loading Model: {folder}')
		folder_path = f'{root_folder}/{folder}/'
		input_path = f'{folder_path}/inputs.npy'
		output_path = f'{folder_path}/outputs.npy'

		if not exists(input_path):
			raise FileNotFoundError(f'File "{input_path}" not found.')
		if not exists(output_path):
			raise FileNotFoundError(f'File "{output_path}" not found.')

		input_data = load(input_path)
		output_data = load(output_path)

		output_data = tile(output_data, (input_data.shape[0] // output_data.shape[0], 1))

		inputs.append(input_data)
		outputs.append(output_data)

	if shuffle_lines:
		return shuffle_data(concatenate(inputs), concatenate(outputs))
	else:
		return concatenate(inputs), concatenate(outputs)


def load_inputs(path: str):
	input_path = f'{path}/inputs.npy'

	if not exists(input_path):
		raise FileNotFoundError(f'File "{input_path}" not found.')

	return load(input_path)


def get_plot_name(path: str = './plots/'):
	if not exists(path):
		makedirs(path)
	plots = [i for i in listdir(path) if isfile(f'{path}/{i}') and i.endswith('.png') and i.startswith('plot')]
	return f'{path}/plot{len(plots)}.png'
