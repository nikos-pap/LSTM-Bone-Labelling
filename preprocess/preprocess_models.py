from sys import argv
from os import listdir, system
from numpy import array, save as npsave
from os.path import isdir, exists


def load_bone_categories(path: str):
	f = open(path, 'r')
	data = f.read().split()
	data = array([[int(j == int(i)) for j in range(5)] for i in data], dtype=float)
	print(data.shape)
	f.close()
	return data


def generate_outputs(root_folder):
	print(f'Generating outputs for "{root_folder}"')
	
	if not exists(root_folder):
		raise FileNotFoundError(f'Folder "{root_folder}" not found.')

	for model_name in listdir(root_folder):
		model_path = f'{root_folder}/{model_name}'
		filepath = f'{model_path}/{model_name}.bones'

		if not isdir(f'{root_folder}/{model_name}') or not exists(filepath):
			continue
		
		print(f'Parsing model: {model_name}...', end=' ')

		data = load_bone_categories(filepath)
		npsave(f'{model_path}/outputs.npy', data)

		print('Done')
	

def main():
	if len(argv) > 1:
		system(f'blender -b -P get_anim.py -- {" ".join(argv[1:])}')
		for path in argv[1:]:
			generate_outputs(path)


if __name__ == '__main__':
	main()
