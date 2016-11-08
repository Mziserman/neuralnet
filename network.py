import numpy as np
import random
from neuron import Neuron
from layer import Layer


class Network:
    def __init__(self, shape):
        self.layers = [];
        for i in range(0, len(shape)):
            self.layers.append(Layer(shape[i]))
        self.connect()
        
    def connect(self):
        for i in range(0, len(self.layers) - 1):
            print(i)
            self.layers[i].connect(self.layers[i + 1])