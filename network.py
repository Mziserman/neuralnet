import numpy as np
import random
from neuron import Neuron
from layer import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self, shape):
        self.layers = [];
        for i in range(0, len(shape)):
            self.layers.append(Layer(shape[i]))
        self.connect()
        
    def connect(self):
        for i in range(0, len(self.layers) - 1):
            self.layers[i].connect(self.layers[i + 1])
    
    def feed_forward(self, input):
        i = 0
        for neuron in self.layers[0].neurons:
            neuron.feed_forward(input[i])
            i += 1
            
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.feed_forward(neuron.data)
                
        ##for neuron in self.layers[-1].neurons:
            ##print(neuron.sigmoid(neuron.data))
            
    def back_propagate(self, targets, N, M):
        if len(targets) != len(self.layers[-1].neurons):
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        i = 0
        for neuron in self.layers[-1].neurons:
            error = targets[i] - neuron.sigmoid(neuron.data)
            neuron.delta = neuron.derivative_sigmoid(neuron.sigmoid(neuron.data)) * error
            i += 1

        # calculate error terms for hiddens
        for i in range(len(self.layers)):
            for neuron in self.layers[-(i+1)].neurons:
                error = 0
                for connection in neuron.connections["backward"]:
                    error = error + neuron.delta * connection["weight"]
                for connection in neuron.connections["backward"]:
                    connection["neuron"].delta = neuron.derivative_sigmoid(neuron.sigmoid(neuron.data)) * error
        
        for i in range(len(self.layers)):
            for neuron in self.layers[-(i+1)].neurons:
                for connection in neuron.connections["backward"]:
                    change = neuron.delta * neuron.sigmoid(connection["neuron"].data)
                    new_weight = connection["weight"] + N * change + M * neuron.change
                    connection["weight"] = new_weight
                    neuron.change = change
                    print(new_weight)
                
                    for connection in connection["neuron"].connections["forward"]:
                        if connection["neuron"] == neuron:
                            connection["weight"] = new_weight
                
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - sigmoid(self.layers[-1].neurons[k].data))**2
        
        self.remove_data()
        
        return error
    
    def remove_data(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.data = 0
    
    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error = error + self.back_propagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)