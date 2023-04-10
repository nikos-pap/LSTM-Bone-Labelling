import bpy
import os
import numpy as np

#os.system('cls')
obj = bpy.data.objects
armature = obj[0]
sce = bpy.context.scene
mesh = []
print(armature.pose.bones[0].matrix_basis[:4][:3])

inputs = []
outputs = []

def flatten_vector_list(vecs):
    result = []
    for vec in vecs:
        result.extend(list(vec))
    return result
        

for f in range(sce.frame_start, 250 + 1):
    sce.frame_set(f)
    m = []
    for o in obj[2:]:
        m.extend([v.co for v in o.data.vertices])
    b = [flatten_vector_list(bone.matrix_basis[:3]) for bone in armature.pose.bones]
    outputs.append(m)
    inputs.append(b)

inputs = np.array(inputs)
outputs = np.array(outputs)
np.save(r'D:\Code\Github\LSTM-motion-transfer\inputs.npy',inputs)
np.save(r'D:\Code\Github\LSTM-motion-transfer\outputs.npy',outputs)
print(inputs.shape)
print(outputs.shape)
