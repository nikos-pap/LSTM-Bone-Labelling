from utils.bone_model import BoneLSTM
from utils.data_manager import load_data
from time import time


def parse_params(params):
    layers: list[list[str]] = []
    i = 0
    while i < len(params):
        if 'lstm' in params[i]:
            layers.append(params[i].split(':'))
            layers[-1][1] = layers[-1][1]
        if 'dropout' in params[i]:
            layers.append(params[i].split(':'))
            layers[-1][1] = layers[-1][1]
        if 'attention' in params[i]:
            layers.append(['attention'])
        if 'dense' in params[i]:
            layers.append(params[i].split(':'))
            layers[-1][1] = layers[-1][1]

        i += 1
    return layers


with open('../test_configs.txt', 'r') as f:
    for line in f:
        test_nns = parse_params(line.split())
        start = time()
        network = BoneLSTM(test_nns)
        
        training_data = load_data('./models/training2/')
        print(f'Training Shapes: {training_data[0].shape}, {training_data[1].shape}')
        network.train(training_data[0], training_data[1], epochs=200)
        
        evaluation_inputs, evaluation_outputs = load_data('./models/test2/', shuffle_lines=False)
        print(f'Shapes: {evaluation_inputs[0].shape}, {evaluation_outputs[0].shape}')
        network.evaluate(evaluation_inputs, evaluation_outputs)
        print(f'Total time: {time() - start} seconds')
