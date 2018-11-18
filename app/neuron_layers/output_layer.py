import numpy as np

from app.neuron_layers.neuron_layer import NeuronLayer
from app.neuron import Neuron


class OutputNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons: int, inputs_layer_size: int):
        super().__init__(num_neurons, inputs_layer_size)
        self._exp_total_inputs = list()
        self._sum_exp_input = None

    def _softmax(self, inputs: list):
        outputs = list()
        for neuron in self._neurons:
            self._exp_total_inputs.append(np.exp(neuron.calculate_total_net_input(inputs)))

        self._sum_exp_input = sum(self._exp_total_inputs)
        for exp_input, neuron in zip(self._exp_total_inputs, self._neurons):
            neuron.output = exp_input / self._sum_exp_input
            outputs.append(neuron.output)

        return outputs

    def update_weights(self, targets: list, hidden_layer_outputs: list, learning_rate: float):
        for neuron, target, input_exp in zip(self._neurons, targets, self._exp_total_inputs):
            error_derivative_to_output, output_derivative_to_input = self._calculate_error(target,
                                                                                           neuron,
                                                                                           input_exp,
                                                                                           len(targets))
            for i in range(len(hidden_layer_outputs)):
                delta_weight = error_derivative_to_output * output_derivative_to_input * hidden_layer_outputs[i].output
                neuron.set_weight(i, neuron.get_weight(i) - (learning_rate * delta_weight))
                pass
            pass
        pass

    def _calculate_error(self, target: int, neuron: Neuron, input_exp: float, layer_size: int):
        error_derivative_to_output = - (target / neuron.output + (1 - target) / (1 - neuron.output)) / layer_size
        output_derivative_to_input = (input_exp * (self._sum_exp_input - input_exp)) / (self._sum_exp_input ** 2)

        return error_derivative_to_output, output_derivative_to_input

    def feed_forward(self, inputs: list):
        outputs = self._softmax(inputs)
        return outputs
