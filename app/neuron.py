import numpy as np


class Neuron:
    def __init__(self, weights: list, activation_func):
        self._weights = weights
        self._activation_func = activation_func
        self._output = None

    def calculate_output(self, inputs: list):
        self._output = self._activation_func(self._calculate_total_net_input(inputs))

        return self._output

    def _calculate_total_net_input(self, inputs: list):
        total = 0
        for input_value, weight in zip(inputs, self._weights):
            total += input_value * weight

        return total
