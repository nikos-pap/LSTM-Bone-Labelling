import bpy
from os import listdir
from bisect import insort
from mathutils import Vector
from sys import path as envpath
envpath.append('..')
from utils.bone_model import BoneLSTM
from preprocess.get_anim import get_bone_positions
from utils.constants import CATEGORY_STARTS


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


def filter_group(model, root_bone, group, filtered):
    i = 0
    print(model.matrix_world)
    right = -(model.matrix_world @ Vector((1,0,0)))
    while i < len(group):
        vec = model.pose.bones[group[i]].center - root_bone.center
        vec.normalize()
        if right.dot(vec) < 0:
            i += 1
        else:
            filtered.append(group.pop(i))


def label(network):
    model = bpy.data.objects[0]

    bones = get_bone_positions()
    groups = network.get_groups(bones)
    for bone in model.pose.bones:
        bone.matrix_basis.identity()

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


def main():
    # path = 'D:\\Code\\Github\\Labelling\\models\\test2\\Mremireh\\'
    path = '../models/test2/Mremireh/'
    files = [path + f for f in listdir(path) if f.endswith('.bvh')]
    end = []
    network = BoneLSTM()
    network.load()
    for file in files:
        load_file(path=file)
        end.append(label(network))
    print(end)
    

main()
