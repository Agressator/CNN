import numpy as np
from app.neuron_layers.neuron_layer import NeuronLayer


class HiddenNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons: int, inputs_layer_size: int):
        super().__init__(num_neurons, inputs_layer_size)

    def _sigmoid(self, arg: float):
        return 1 / (1 + np.exp(-arg))

    def feed_forward(self, inputs: list):
        outputs = list()
        for neuron in self._neurons:
            total_input = neuron.calculate_total_net_input(inputs)
            neuron.output = self._sigmoid(total_input)
            outputs.append(neuron.output)

        return outputs
