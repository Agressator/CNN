
class Neuron:
    def __init__(self, weights: list):
        self._weights = weights
        self.output = None

    def calculate_total_net_input(self, inputs: list):
        total = 0
        for input_value, weight in zip(inputs, self._weights):
            total += input_value * weight

        return total
