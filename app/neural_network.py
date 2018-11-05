from app.neuron_layer import NeuronLayer
from app.activation_functions import sigmoid, softmax

INPUT_SIZE = 784
OUTPUT_SIZE = 10


class NeuralNetwork:
    def __init__(self, hidden_layer_size: int):
        self._hidden_layer = NeuronLayer(hidden_layer_size, INPUT_SIZE, sigmoid)
        # TODO: send softmax func as a parameter trough the NeuronLayer constructor
        self._output_layer = NeuronLayer(OUTPUT_SIZE, hidden_layer_size, sigmoid)

    def _feed_forward(self, inputs: list):
        hidden_layer_outputs = self._hidden_layer.feed_forward(inputs)
        return self._output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs: list, training_outputs: list):
        self._feed_forward(training_inputs)
