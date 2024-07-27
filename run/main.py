import os
import numpy as np
from sys import path as envpath
envpath.append('../utils/')
from data_manager import load_inputs
from bone_model import BoneLSTM


end = []
network = BoneLSTM()
network.load()

path = '../models/test2/Mremireh/'
# load data
inputs = load_inputs(path)
#run NN
outs = network.predict(inputs)
print(outs.shape)
print(outs)
np.save(f'{path}results.npy', outs)
#run get labels
# os.system(f'blender -b -P bone_label.py -- {path}/Mremireh.bvh')