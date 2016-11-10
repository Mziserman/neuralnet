import numpy as np
import random

class Neuron:
    def __init__(self):
        self.connections = {
            "forward": [],
            "backward": []
        }
        self.data = 0
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
        for connection in self.connections["forward"]:
            forward_data = self.sigmoid(input * connection["weight"])
            ##backward_connection = next(x for x in connection["neuron"].connections["backward"] if self == x["neuron"])
            connection["neuron"].data += forward_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def derivative_sigmoid(self, x):
        return x * (1 - x)
    