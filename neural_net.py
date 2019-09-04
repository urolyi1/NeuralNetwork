import numpy as np
np.random.seed(42)
DEFUALT_ETA = 0.1

def sigmoid(x):
    return 1. / (1. + np.exp(-1. * x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
        self.weights = []
        prev_dim = num_nodes[0]
        for layer in num_nodes[1:]:
            self.weights.append(np.random.rand(prev_dim, layer))
            #print("({}, {})".format(prev_dim, layer))
            prev_dim = layer

    def feedforward(self, init_input):
        # output_val = self.trans_func(activation_val)
        self.outputs = []
        self.outputs.append(init_input)
        self.activations = []
        prev_input = init_input
        for idx, layer in enumerate(self.weights):
            activation_val = np.dot(prev_input, layer)
            self.activations.append(activation_val)
            prev_input = self.trans_func(self.activations[idx])
            self.outputs.append(prev_input)
        return prev_input

    def backprop(self, y, num_samples):
        iterable = enumerate(zip(reversed(self.outputs), reversed(self.activations)))
        for idx, (output_val, activation_val) in iterable:
            layer_idx = self.num_layers - idx - 1
            #print("Layer idx: {}".format(layer_idx))
            # Last Layer
            if layer_idx == self.num_layers - 1:
                #print("Last layer")
                dz = output_val - y
                #print("dz: {}".format(dz))
            else:
                #print("Hidden Layer")
                dz = np.multiply(np.dot(dz, self.weights[layer_idx].T), d_sigmoid(self.activations[layer_idx - 1]))

            #print("Shape 1: {}".format(self.outputs[layer_idx - 1].T.shape))
            #print("Shape 2: {}".format(dz.shape))
            dw = np.dot(self.outputs[layer_idx - 1].T, dz) / num_samples
            #print('dw shape: {}'.format(dw.shape))
            #print("a: {}".format(self.outputs[layer_idx].T))
            #print(dw * self.learn_rate)
            #print(self.weights[layer_idx])
            #print('weights shape: {}'.format(self.weights[layer_idx - 1].shape))
            self.weights[layer_idx - 1] -= dw * self.learn_rate

    def get_loss(self, y, y_hat):
        indiv_loss = 0.5 * np.power(y - y_hat , 2)
        return np.mean([np.linalg.norm(indiv_loss[row, :], 2) for row in range(indiv_loss.shape[0])])

    def dz_from_loss(self, y, y_hat):
        return y_hat - y

test = NeuralNet([2, 20, 20, 3])
x = np.random.rand(10000, 2)
epochs = 100
y = np.concatenate(([(x[:,0] + x[:,1])], [(x[:,0] - x[:,1])], np.abs([(x[:,0] - x[:,1])]))).T
for i in range(1, epochs):
    y_hat = test.feedforward(x)
    print(test.get_loss(y, y_hat))
    test.backprop(y, 10000)
    y_hat = test.feedforward(x)
    print(test.get_loss(y, y_hat))
x = np.random.rand(10000, 2)
y = np.concatenate(([(x[:,0] + x[:,1])], [(x[:,0] - x[:,1])], np.abs([(x[:,0] - x[:,1])]))).T
y_hat = test.feedforward(x)
print(test.get_loss(y, y_hat))