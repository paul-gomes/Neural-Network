import random
from activationFunc import ActivationFunction
from utils import draw_progess_bar
import numpy as np

class NeuralNetwork:
    def __init__(self, activ_func = 'Sigmoid', learning_rate='0.01', debug=True):
        self.activ_func = ActivationFunction(types=activ_func)
        self.layers = []
        self.learning_rate = learning_rate
        self.debug = debug
        
    def add_layer(self, n_inputs, n_neurons):
        layer = NeuralLayer(n_inputs, n_neurons, self.activ_func)
        self.layers.append(layer)
        return None
               
    def feed_forward(self, inputs):
        for i, layer in enumerate(self.layers):
            inputs = layer.feed_forward(inputs) # next_input = previous_output

            if self.debug:
                print('Layer {}, Output: {}'.format(i+1, inputs))

        return inputs
        
    def feed_backward(self, targets): # backpropagating
        if len(targets) != len(self.layers[-1].neurons):
            raise Exception('wrong target numbers')

        # calculate deltas of output layer
        for j, neuron_j in enumerate(self.layers[-1].neurons):
            error = - (targets[j] - neuron_j.output)
            neuron_j.calculate_delta(error)

        if self.debug:
            print("Output_Layer: deltas: {}".format(self.layers[-1].get_deltas()))


        # calculate the hidden layers
        n_hidden_layers = len(self.layers[:-1])
        l = n_hidden_layers - 1

        while l >= 0:
            curr_layer, last_layer = self.layers[l], self.layers[l+1]

            for i, neuron_i in enumerate(curr_layer.neurons):
                # sum up the errors sent from the last layer
                total_error = 0
                for j, neuron_j in enumerate(last_layer.neurons):
                    total_error += neuron_j.delta * neuron_j.weights[i] # total_error += delta_j * input_i_to_j

                neuron_i.calculate_delta(total_error)

            if self.debug:
                print("Layer {}: deltas: {}".format(l+1, curr_layer.deltas))

            l -= 1

        return None
        
        
    def update_weights(self):
        learning_rate = self.learning_rate
        for l in self.layers:
            l.update_weights(learning_rate)

        return None
    
    
    def calculate_single_error(self, targets, actual_outputs):
        error = 0
        for i in range(len(targets)):
            error += (targets[i] - actual_outputs[i]) ** 2
        return error
    
    def calculate_total_error(self, dataset, targets):
        """
        Return mean squared error of dataset
        """
        total_error = 0
        for count, inputs in enumerate(dataset):
            actual_outputs = self.feed_forward(inputs)  
            total_error += self.calculate_single_error(targets[count], actual_outputs)

        return total_error / len(dataset)
    
    def train(self, dataset, targets, n_iterations=100, print_error_report=True):

        print('\n> Training...')

        for i in range(n_iterations):
            print('| # {}/{}\t| '.format(i+1, n_iterations), end="", flush=True)
            for count, value in enumerate(dataset):
                if self.debug:
                    print('\n>>> data #{}'.format(count+1))
                self.feed_forward(value)
                self.feed_backward(targets[count])
                self.update_weights()
            total_error = self.calculate_total_error(dataset, targets)

            if print_error_report:
                print(' Total error: {}'.format(total_error))
            else:
                draw_progess_bar(n_finished=i+1, n_jobs=n_iterations, sleep_time=0.05) # draw progress bar

        print('Training Finish. Error = {}\n'.format(total_error))

        return None
    
    def test(self, dataset, targets):
        print('\n> Testing...')
        accurateCount = 0
        for count, value in enumerate(dataset):
            if self.debug:
                print('\n>>> data #{}'.format(count+1))

            actual_outputs = self.feed_forward(value)
            pred_value_index = actual_outputs.index(max(actual_outputs))
            target_value_index = np.where((targets[count]) == 1)[0][0]
            if(pred_value_index == target_value_index):
                accurateCount +=1
            
            print('[#{}] {} -> {} (targets={}) {}, {}'.format(count, value, actual_outputs, targets[count], pred_value_index, target_value_index))
        total_error = self.calculate_total_error(dataset, targets)
        accuracy = accurateCount/(count+1)

        print('Testing Finish. Error: {}\n'.format(total_error))
        print('Testing accuracy: {}\n'.format(accuracy))

        return None
    
    
    
    
class NeuralLayer:
    __counter = 0
    def __init__(self, n_inputs, n_neurons, activ_func):
        self.__counter = NeuralLayer.__counter = NeuralLayer.__counter + 1
        self.__neurons = [Neuron(n_inputs, activ_func) for _ in range(n_neurons)]

    @property
    def neurons(self):
        return self.__neurons

    # @property
    # def actual_outputs(self):
    #     return [neuron.output for neuron in self.neurons]

    @property
    def deltas(self):
        return [i.delta for i in self.neurons]

    def feed_forward(self, inputs):
        return [neuron.calculate_output(inputs) for neuron in self.neurons]

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
        return None

    def __str__(self):
        return '-- Layer {}  # of neurons: {}'.format(self.__counter, len(self.neurons))
    

class Neuron:
    def __init__(self, n_weights, activ_func='Sigmoid', bias=1):
        self.__weights = [random.random() for i in range(n_weights)]
        self.__bias = bias
        self.__output = 0.0
        self.__inputs = []
        self.__delta = 0.0
        self.__n_weights = n_weights
        self.__activation = activ_func

    @property
    def output(self):
        return self.__output
    @property
    def delta(self):
        return self.__delta
    @property
    def weights(self):
        return self.__weights

    def calculate_output(self, inputs):
        n_weights = self.__n_weights
        if len(inputs) != n_weights:
            raise Exception('wrong inputs number')

        output = sum([inputs[i] * self.__weights[i] for i in range(n_weights)])
        a_output = self.__activation.func(output + self.__bias)

        # set the variables
        self.__inputs = inputs
        self.__output = a_output

        return a_output

    def calculate_delta(self, error):
        self.__delta = error * self.__activation.dfunc(self.__output)

    def update_weights(self, learning_rate):
        for i in range(self.__n_weights):
            self.__weights[i] -= learning_rate * self.__delta * self.__inputs[i]
        self.__bias -= learning_rate * self.__delta

        # update output
        self.calculate_output(self.__inputs)

        return None

    def __str__(self):
        return '--- weights = {}, bias = {}'.format(self.__weights, self.__bias)