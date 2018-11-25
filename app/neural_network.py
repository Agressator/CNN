import numpy as np

from app.neuron_layers.hidden_layer import HiddenNeuronLayer
from app.neuron_layers.output_layer import OutputNeuronLayer

INPUT_SIZE = 784
OUTPUT_SIZE = 10


class NeuralNetwork:
    def __init__(self, hidden_layer_size: int, learning_rate: float):
        self._layers = [
            HiddenNeuronLayer(hidden_layer_size, INPUT_SIZE),
            OutputNeuronLayer(OUTPUT_SIZE, hidden_layer_size)
        ]
        self._learning_rate = learning_rate

    def _feed_forward(self, inputs: list):
        current_inputs = inputs
        for layer in self._layers:
            current_inputs = layer.feed_forward(current_inputs)

        return current_inputs

    def _cross_entropy_error(self, computed: list, targets: list):
        total_error = 0
        for neuron_output, expected_output in zip(computed, targets):
            error = expected_output * np.log(neuron_output) + (1 - expected_output) * np.log(1 - neuron_output)
            total_error += error

        return -total_error/len(computed)

    def train(self, training_inputs: list, training_outputs: list):
        output = self._feed_forward(training_inputs)
        total_error = self._cross_entropy_error(output, training_outputs)

        error_derivatives_to_input = self._layers[1].update_weights(training_outputs,
                                                                    self._layers[0].neurons,
                                                                    self._learning_rate)

        # hidden_neurons_weights = list()
        # for hidden_neuron_num in range(len(self._layers[0].neurons)):
        #     hidden_neurons_weights.append(
        #         [output_neurons.get_weight(hidden_neuron_num) for output_neurons in self._layers[1].neurons]
        #     )
        # error_derivatives_to_hidden_output = list()
        # for hidden_neuron in hidden_neurons_weights:
        #     error_derivative_to_hidden_output = 0
        #     for error_derivative_to_input, weights in zip(error_derivatives_to_input, hidden_neuron):
        #         error_derivative_to_hidden_output += error_derivative_to_input * weights
        #     error_derivatives_to_hidden_output.append(error_derivative_to_hidden_output)

        self._layers[0].update_weights(error_derivatives_to_input, self._layers[1].neurons, self._learning_rate)
        pass
