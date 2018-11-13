import numpy as np
from app.neuron_layers.neuron_layer import NeuronLayer


class OutputNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons: int, inputs_layer_size: int):
        super().__init__(num_neurons, inputs_layer_size)

    def _softmax(self, inputs: list):
        outputs = list()
        exp_total_inputs = list()
        for neuron in self._neurons:
            exp_total_inputs.append(np.exp(neuron.calculate_total_net_input(inputs)))

        sum_exp_input = sum(exp_total_inputs)
        for exp_input, neuron in zip(exp_total_inputs, self._neurons):
            neuron.output = exp_input / sum_exp_input
            outputs.append(neuron.output)

        return outputs

    def feed_forward(self, inputs: list):
        outputs = self._softmax(inputs)
        return outputs
