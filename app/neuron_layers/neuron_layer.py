import numpy as np
from app.neurons.sigmoid_neuron import Neuron


class NeuronLayer:
    def __init__(self, num_neurons: int, inputs_layer_size: int, activation_func):
        self._neurons = []
        for count in range(num_neurons):
            weights = self._init_weights_from_inputs_to_current_layer_neurons(inputs_layer_size)
            self._neurons.append(Neuron(weights, activation_func))

    def _init_weights_from_inputs_to_current_layer_neurons(self, inputs_neurons_count: int, weights: list=None):
        if weights and inputs_neurons_count == len(weights):
            return weights

        return 2 * np.random.rand(inputs_neurons_count) - 1

    def feed_forward(self, inputs: list):
        outputs = list()
        for neuron in self._neurons:
            outputs.append(neuron.calculate_output(inputs))

        return outputs


if __name__ == "__main__":
    nl = NeuronLayer(3, 2)
