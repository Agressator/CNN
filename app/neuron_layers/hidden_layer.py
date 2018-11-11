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

    def _sigmoid(self, arg: float):
        return 1 / (1 + np.exp(-arg))

    def feed_forward(self, inputs: list):
        outputs = list()
        for neuron in self._neurons:
            total_input = neuron.calculate_total_net_input(inputs)
            neuron.output = self._sigmoid(total_input)
            outputs.append(neuron.output)

        return outputs
