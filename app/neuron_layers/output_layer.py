import numpy as np
from app.neuron import Neuron


class NeuronLayer:
    def __init__(self, num_neurons: int, inputs_layer_size: int):
        self._neurons = []
        for count in range(num_neurons):
            weights = self._init_weights_from_inputs_to_current_layer_neurons(inputs_layer_size)
            self._neurons.append(Neuron(weights))

    def _init_weights_from_inputs_to_current_layer_neurons(self, inputs_neurons_count: int, weights: list=None):
        if weights and inputs_neurons_count == len(weights):
            return weights

        return 2 * np.random.rand(inputs_neurons_count) - 1

    def _softmax(self, arg, inputs):
        exp_input = [np.exp(input_value) for input_value in inputs]
        return np.exp(arg) / sum(exp_input)

    def feed_forward(self, inputs: list):
        outputs = list()
        total_inputs = list()
        for neuron in self._neurons:
            total_inputs.append((neuron.calculate_total_net_input(inputs), neuron))

        for total_input in total_inputs:
            neuron.output = self._softmax(total_input, total_inputs)
            outputs.append(neuron.output)

        return outputs
