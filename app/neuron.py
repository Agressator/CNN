
class Neuron:
    def __init__(self, weights: list):
        self._weights = weights
        self._output = None

    def get_output(self):
        return self._output

    def set_output(self, value: float):
        self._output = value

    output = property(get_output, set_output)

    def get_weight(self, number: int):
        return self._weights[number]

    def set_weight(self, number: int, value: float):
        self._weights[number] = value

    def calculate_total_net_input(self, inputs: list):
        total = 0
        for input_value, weight in zip(inputs, self._weights):
            total += input_value * weight

        return total
