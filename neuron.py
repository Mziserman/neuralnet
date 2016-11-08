import numpy as np
import random

class Neuron:
    def __init__(self):
        self.connections = []
        
    def connect(self, layer):
        for neuron in layer.neurons:
            self.connections.append({
                "neuron": neuron,
                "weight": random.uniform(-1, 1)
            })

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))