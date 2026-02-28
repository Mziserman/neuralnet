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
        # Clear previous data before forward pass
        self.remove_data()

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
            error = targets[i] - neuron.output
            neuron.delta = neuron.derivative_sigmoid(neuron.output) * error
            i += 1

        # calculate error terms for hiddens
        # Go from second-to-last layer backwards to first hidden layer
        for i in range(len(self.layers) - 2, 0, -1):
            for neuron in self.layers[i].neurons:
                error = 0
                # Sum errors from all neurons in the NEXT layer (forward connections)
                for connection in neuron.connections["forward"]:
                    error += connection["neuron"].delta * connection["weight"]
                # Set delta on CURRENT neuron
                neuron.delta = neuron.derivative_sigmoid(neuron.output) * error

        # Update weights - iterate through all layers (except output) and update forward connections
        for i in range(len(self.layers) - 1):
            for neuron in self.layers[i].neurons:
                for connection in neuron.connections["forward"]:
                    # weight update: destination_delta * source_output
                    change = connection["neuron"].delta * neuron.output

                    # Initialize change in connection if not exists
                    if "change" not in connection:
                        connection["change"] = 0

                    new_weight = connection["weight"] + N * change + M * connection["change"]
                    connection["weight"] = new_weight
                    connection["change"] = change

                    # Update the corresponding backward connection to keep in sync
                    for backward_conn in connection["neuron"].connections["backward"]:
                        if backward_conn["neuron"] == neuron:
                            backward_conn["weight"] = new_weight
                            backward_conn["change"] = change
                
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.layers[-1].neurons[k].output)**2
        
        self.remove_data()
        
        return error
    
    def remove_data(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.data = 0
                neuron.output = 0
    
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