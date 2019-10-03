import numpy as np

#default step_size
DEFUALT_ETA = 0.1


# Transition Functions and their derivatives
def sigmoid(x):
    return 1. / (1. + np.exp(-1. * x))
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

class NeuralNet:
    def __init__(self, num_nodes, num_hidden_layers=1, 
                 learning_rate=DEFUALT_ETA, transition_function=sigmoid,
                 derivative_transition=d_sigmoid):
        self.learn_rate= learning_rate
        self.num_hidden = num_hidden_layers
        self.trans_func = transition_function
        self.num_layers = len(num_nodes)
        self.initialize_weights(num_nodes)

    def initialize_weights(self, num_nodes):
        ''' Generate weights matrices for each layer randomly'''
        self.weights = []
        self.bias = []
        prev_dim = num_nodes[0]
        for layer in num_nodes[1:]:
            self.weights.append(np.random.rand(prev_dim, layer))
            self.bias.append(np.zeros((1, layer)))
            prev_dim = layer

    def feedforward(self, init_input):
        ''' Generate output from init_input by putting through layers'''
        self.outputs = []
        self.outputs.append(init_input)
        self.activations = []
        prev_input = init_input

        # for each layer
        for idx, layer in enumerate(self.weights):
            # layer's output = transition_func(prev_input * weights) + bias
            activation_val = np.add(np.dot(prev_input, layer), self.bias[idx])
            self.activations.append(activation_val)
            prev_input = self.trans_func(self.activations[idx])
            self.outputs.append(prev_input)
        return prev_input

    def backprop(self, y, num_samples):
        ''' Optimize weights through backpropagation '''

        # iterate over outputs and activation input values
        iterable = enumerate(zip(reversed(self.outputs), reversed(self.activations)))
        for idx, (output_val, activation_val) in iterable:
            layer_idx = self.num_layers - idx - 1

            # if last layer dz = y_hat - y else previous dz * weights * derivative of transition function
            if layer_idx == self.num_layers - 1:
                dz = output_val - y
            else:
                dz = np.multiply(np.dot(dz, self.weights[layer_idx].T), d_sigmoid(self.activations[layer_idx - 1]))

            # Changing weights in direction of gradient
            dw = np.dot(self.outputs[layer_idx - 1].T, dz) / num_samples
            self.weights[layer_idx - 1] -= dw * self.learn_rate
            self.bias[layer_idx - 1] = np.subtract(self.bias[layer_idx - 1], np.sum(dz, axis=0, keepdims=True) / num_samples)

    def get_loss(self, y, y_hat):
        indiv_loss = 0.5 * np.power(y - y_hat , 2)
        return np.mean([np.linalg.norm(indiv_loss[row, :], 2) for row in range(indiv_loss.shape[0])])

    def train(self, num_iterations, train_x, train_y, test_x, test_y):
        init_out = self.feedforward(test_x)
        print(self.get_loss(test_y, init_out))
        for i in range(num_iterations):
            y_hat = self.feedforward(train_x)
            self.backprop(train_y, train_x.shape[0])

        optimized_out = self.feedforward(test_x)
        print(self.get_loss(test_y, optimized_out))
