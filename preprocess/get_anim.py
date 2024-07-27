import bpy
from os import listdir
from os.path import exists
from sys import argv, path as envpath
from numpy import concatenate, array, save as npsave
envpath.append('..')
from utils.constants import FRAME_NUM


def get_bone_positions():
	ob = bpy.data.objects[0]
	sce = bpy.context.scene
	frames = []

	for f in range(sce.frame_start, FRAME_NUM + 1):
		sce.frame_set(f)
		frames.append([bone.center for bone in ob.pose.bones])

	bones = [[frame[i] for frame in frames] for i in range(len(ob.pose.bones))]

	result = array(bones, dtype=float)
	return result


def save_bones(bones, filename):
	npsave(filename, bones)
	print(f'File Saved at {filename}')


def load_file(path):
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete()
	bpy.ops.import_anim.bvh(filepath=path, axis_forward='-Z', axis_up='Y', filter_glob="*.bvh",
							target='ARMATURE', global_scale=1.0, frame_start=1, use_fps_scale=False,
							update_scene_fps=False,	update_scene_duration=False, use_cyclic=False, rotate_mode='NATIVE')


def parse_models(model_path):

	if not exists(model_path):
		raise FileNotFoundError(f'Model folder ({model_path}) not found.')

	for folder in listdir(model_path):
		folder_path = f'{model_path}/{folder}'
		files = [i for i in listdir(folder_path) if i.endswith('.bvh')]
		bones = []

		for bvh in files:
			file_path = f'{folder_path}/{bvh}'

			if not exists(file_path) or not file_path.endswith('.bvh'):
				continue

			load_file(file_path)  # load bvh
			print(f'File \'{bvh}\' loaded.')
			bones.append(get_bone_positions())  # get position matrices
		file_path = f'{folder_path}/inputs.npy'

		save_bones(concatenate(bones), file_path)  # save data in file


def get_inputs():
	try:
		index = argv.index('--') + 1
	except ValueError:
		return None
	return argv[index:]


def main():
	inputs = get_inputs()
	if inputs is None:
		print('No input folders given.')
		return

	for path in inputs:
		parse_models(path)


if __name__ == '__main__':
	main()
