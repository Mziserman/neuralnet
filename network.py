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
            self.layers[i].connect(self.layers[i + 1])
            
    def input_data(self, data):
        self.data = data
        
    def predict(self, data):
        for datum in data:
            for i in range(0, len(datum)):
                feature = datum[i]
    
    def feed_forward(self, input):
        i = 0
        for neuron in self.layers[0].neurons:
            neuron.feed_forward(input[i])
            i += 1
            
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.feed_forward(neuron.data)