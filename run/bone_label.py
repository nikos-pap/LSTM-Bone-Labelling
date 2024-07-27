import bpy
from os import listdir
from os.path import dirname
import numpy as np
from bisect import insort
from mathutils import Vector
from sys import path as envpath, argv
envpath.append('../utils/')
from constants import CATEGORY_STARTS
from data_manager import argmax


def load_file(path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_anim.bvh(filepath=path, axis_forward='-Z', axis_up='Y', 
                            filter_glob="*.bvh", target='ARMATURE', global_scale=1.0,
                            frame_start=1, use_fps_scale=False, update_scene_fps=False,
                            update_scene_duration=False, use_cyclic=False, rotate_mode='NATIVE')


def dfs(node, lvl):
    result = [lvl]
       
    if len(node.children) == 0:
        return result
       
    for bone in node.children:
        result += dfs(bone, lvl + 1)
       
    return result

def load_groups(path, bone_num):
    folder = dirname(path)
    results = np.load(f'{folder}/results.npy')
    groups = [[] for _ in range(5)]

    for index, outputs in enumerate(results[:bone_num]):
        groups[argmax(outputs)].append(index)

    groups.insert(4, [])
    groups.insert(3, [])

    return groups


def filter_group(model, root_bone, group, filtered):
    i = 0

    while i < len(group):
    
        if model.pose.bones[group[i]].center.x > root_bone.center.x:
            i += 1
        else:
            filtered.append(group.pop(i))


def label(path):
    model = bpy.data.objects[0]

    groups = load_groups(path, bone_num=len(model.data.bones))
    model.data.pose_position = 'REST'
    bpy.context.view_layer.update()

    root_bone = None

    for bone in model.data.bones:
        if bone.parent is None:
            root_bone = bone
            break

    filter_group(model, root_bone, groups[2], groups[3])
    filter_group(model, root_bone, groups[4], groups[5])

    bones = dfs(root_bone, 0)

    end = [0] * len(bones)
    for group, start in zip(groups, CATEGORY_STARTS):
        result = []
        for index in group:
            insort(result, index, key=lambda a: bones[a])
        for i, index in enumerate(result):
            end[index] = start + i
    return end


def get_inputs():
    try:
        index = argv.index('--') + 1
    except ValueError:
        return []
    return argv[index:]


def main():
    inputs = get_inputs()

    if len(inputs) > 0:
        path = inputs[0]
        files = [path]
    else:
        path = '../models/test2/Mremireh/'
        files = [path + f for f in listdir(path) if f.endswith('.bvh')]
    end = []

    for file in files:
        load_file(path=file)
        end.append(label(file))

    print(end)
    

main()
