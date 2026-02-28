import numpy as np
import random

class Neuron:
    def __init__(self):
        self.connections = {
            "forward": [],
            "backward": []
        }
        self.data = 0
        self.output = 0  # Store activated output for backprop
        self.change = 0

    def __str__(self):
        return "data: {0}, forward connections: {1}, backward connection: {2}".format(self.data, len(self.connections["forward"]), len(self.connections["backward"]))
        
    def connect(self, layer):
        for neuron in layer.neurons:
            weight = random.uniform(-1, 1)
            self.connections["forward"].append({
                "neuron": neuron,
                "weight": weight
            })
            neuron.connections["backward"].append({
                "neuron": self,
                "weight": weight
            })

    def feed_forward(self, input):
        if self.connections["backward"]:
            # Hidden/output layers: sum weighted inputs, then sigmoid
            weighted_sum = 0
            for connection in self.connections["backward"]:
                weighted_sum += connection["neuron"].output * connection["weight"]
            self.output = self.sigmoid(weighted_sum)
        else:
            # Input layer: no backward connections, just store the input
            self.output = input

        # Pass the activated value to all forward connections
        for connection in self.connections["forward"]:
            connection["neuron"].data += self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def derivative_sigmoid(self, x):
        return x * (1 - x)
    