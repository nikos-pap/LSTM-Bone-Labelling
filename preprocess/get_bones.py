import bpy
from sys import argv
import os


def load_file(path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.import_anim.bvh(filepath=path, axis_forward='-Z', axis_up='Y', filter_glob="*.bvh",
                            target='ARMATURE', global_scale=1.0, frame_start=1, use_fps_scale=False,
                            update_scene_fps=False, update_scene_duration=False, use_cyclic=False, rotate_mode='NATIVE')


def get_inputs():
    try:
        index = argv.index('--') + 1
    except ValueError:
        return None
    return argv[index:]


def main():
    folders = get_inputs()
    if folders is None:
        print('No input folders given.')
        return
        
    for folder in folders:
        for model_path in os.listdir(folder):

            model_name = os.path.basename(model_path)
            load_file(f'{folder}/{model_path}/{model_name}.bvh')
            bone_names = '\n'.join((bone.name for bone in bpy.data.objects[0].data.bones))
            
            with open(f'{folder}/{model_path}/numbers.txt', 'w') as f:
                f.write(bone_names)


if __name__ == '__main__':
    main()
