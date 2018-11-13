from app.neuron_layers.hidden_layer import HiddenNeuronLayer
from app.neuron_layers.output_layer import OutputNeuronLayer

INPUT_SIZE = 784
OUTPUT_SIZE = 10


class NeuralNetwork:
    def __init__(self, hidden_layer_size: int):
        self._hidden_layer = HiddenNeuronLayer(hidden_layer_size, INPUT_SIZE)
        self._output_layer = OutputNeuronLayer(OUTPUT_SIZE, hidden_layer_size)

    def _feed_forward(self, inputs: list):
        hidden_layer_outputs = self._hidden_layer.feed_forward(inputs)
        output_layer_outputs = self._output_layer.feed_forward(hidden_layer_outputs)

        return output_layer_outputs

    def train(self, training_inputs: list, training_outputs: list):
        self._feed_forward(training_inputs)
