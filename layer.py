import numpy as np
import random
from neuron import Neuron


class Layer:
    
    def __init__(self, neuron_count):
        self.neurons = []
        for i in range(0, neuron_count):
            self.neurons.append(Neuron())
            
    def connect(self, layer):
        for neuron in self.neurons:
            neuron.connect(layer)