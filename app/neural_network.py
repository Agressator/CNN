from app.neuron_layers.hidden_layer import HiddenNeuronLayer
from app.neuron_layers.output_layer import OutputNeuronLayer

INPUT_SIZE = 784
OUTPUT_SIZE = 10


class NeuralNetwork:
    def __init__(self, hidden_layer_size: int):
        self._layers = [
            HiddenNeuronLayer(hidden_layer_size, INPUT_SIZE),
            OutputNeuronLayer(OUTPUT_SIZE, hidden_layer_size)
        ]

    def _feed_forward(self, inputs: list):
        current_inputs = inputs
        for layer in self._layers:
            current_inputs = layer.feed_forward(current_inputs)

        return current_inputs

    def train(self, training_inputs: list, training_outputs: list):
        self._feed_forward(training_inputs)
