import numpy as np

from app.neuron import Neuron
from app.neuron_layers.neuron_layer import NeuronLayer


class HiddenNeuronLayer(NeuronLayer):
    def __init__(self, num_neurons: int, inputs_layer_size: int):
        super().__init__(num_neurons, inputs_layer_size)
        self._total_inputs = list()

    def _sigmoid(self, arg: float):
        return 1 / (1 + np.exp(-arg))

    def feed_forward(self, inputs: list):
        outputs = list()
        for neuron in self._neurons:
            total_input = neuron.calculate_total_net_input(inputs)
            self._total_inputs.append(total_input)
            neuron.output = self._sigmoid(total_input)
            outputs.append(neuron.output)

        return outputs

    def update_weights(self, error_derivatives_to_output_input: list, output_neurons: list, learning_rate: float):
        for neuron_num in range(len(self._neurons)):
            error_derivative_to_hidden_output, hidden_output_derivative_to_hidden_input = \
                self._calculate_error(output_neurons,
                                      error_derivatives_to_output_input,
                                      neuron_num,
                                      self._total_inputs[neuron_num])
            pass
        pass

    # TODO: rename method
    def _calculate_error(self,
                         output_neurons: list,
                         error_derivatives_to_input: list,
                         hidden_neuron_num: int,
                         hidden_input: float):
        hidden_output_derivative_to_hidden_input = self._sigmoid(hidden_input) * (1 - self._sigmoid(hidden_input))

        hidden_neuron_weights = [output_neurons.get_weight(hidden_neuron_num) for output_neurons in output_neurons]
        error_derivative_to_hidden_output = 0
        for error_derivative_to_input, weight in zip(error_derivatives_to_input, hidden_neuron_weights):
            error_derivative_to_hidden_output += error_derivative_to_input * weight

        return error_derivative_to_hidden_output, hidden_output_derivative_to_hidden_input
