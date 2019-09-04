
import numpy as np 
np.random.seed(42) # for reproducibility
X_num_row, X_num_col = [2, 10000] # Row is no. of feature, col is no. of datum points
X_raw = np.random.rand(X_num_row,X_num_col) * 100
print(X_raw)
print("Input Shape: {}".format(X_raw.shape))
y_raw = np.concatenate(([(X_raw[0,:] + X_raw[1,:])], [(X_raw[0,:] - X_raw[1,:])], np.abs([(X_raw[0,:] - X_raw[1,:])])))
# for input a and b, output is a+b; a-b and |a-b|
y_num_row, y_num_col = y_raw.shape
print(y_raw)
print("Input Shape: {}".format(y_raw.shape))
train_ratio = 0.7
num_train_datum = int(train_ratio*X_num_col)
X_raw_train = X_raw[:,0:num_train_datum]
X_raw_test = X_raw[:,num_train_datum:]

y_raw_train = y_raw[:,0:num_train_datum]
y_raw_test = y_raw[:,num_train_datum:]

class scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
def get_scaler(row):
    mean = np.mean(row)
    std = np.std(row)
    return scaler(mean, std)

def standardize(data, scaler):
        return (data - scaler.mean) / scaler.std

def unstandardize(data, scaler):
    return (data * scaler.std) + scaler.mean


# Construct scalers from training set

X_scalers = [get_scaler(X_raw_train[row,:]) for row in range(X_num_row)]
X_train = np.array([standardize(X_raw_train[row,:], X_scalers[row]) for row in range(X_num_row)])

y_scalers = [get_scaler(y_raw_train[row,:]) for row in range(y_num_row)]
y_train = np.array([standardize(y_raw_train[row,:], y_scalers[row]) for row in range(y_num_row)])
# Apply those scalers to testing set

X_test = np.array([standardize(X_raw_test[row,:], X_scalers[row]) for row in range(X_num_row)])
y_test = np.array([standardize(y_raw_test[row,:], y_scalers[row]) for row in range(y_num_row)])


# Check if data has been standardized

print([X_train[row,:].mean() for row in range(X_num_row)]) # should be close to zero
print([X_train[row,:].std() for row in range(X_num_row)])  # should be close to one

print([y_train[row,:].mean() for row in range(y_num_row)]) # should be close to zero
print([y_train[row,:].std() for row in range(y_num_row)])  # should be close to one

class layer:
    def __init__(self, layer_index, is_output, input_dim, output_dim, activation):
        self.layer_index = layer_index # zero indicates input layer
        self.is_output = is_output # true indicates output layer, false otherwise
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # the multiplication constant is sorta arbitrary
        if layer_index != 0:
            self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim) 
            self.b = np.random.randn(output_dim, 1) * np.sqrt(2/input_dim)
            
            
# Change layers_dim to configure your own neural net!
print('input: {}'.format(X_num_row))
print('output: {}'.format(y_num_row))                       
layers_dim = [X_num_row, 4, 4, y_num_row] # input layer --- hidden layers --- output layers
neural_net = []
# Construct the net layer by layer
for layer_index in range(len(layers_dim)):
    if layer_index == 0: # if input layer
        neural_net.append(layer(layer_index, False, 0, layers_dim[layer_index], 'irrelevant'))
    elif layer_index+1 == len(layers_dim): # if output layer
        neural_net.append(layer(layer_index, True, layers_dim[layer_index-1], layers_dim[layer_index], activation='linear'))
    else: 
        neural_net.append(layer(layer_index, False, layers_dim[layer_index-1], layers_dim[layer_index], activation='relu'))
        
        
# Simple check on overfitting

pred_n_param = sum([(layers_dim[layer_index]+1)*layers_dim[layer_index+1] for layer_index in range(len(layers_dim)-1)])
act_n_param = sum([neural_net[layer_index].W.size + neural_net[layer_index].b.size for layer_index in range(1,len(layers_dim))])
print(f'Predicted number of hyperparameters: {pred_n_param}')
print(f'Actual number of hyperparameters: {act_n_param}')
print(f'Number of data: {X_num_col}')

if act_n_param >= X_num_col:
    raise Exception('It will overfit.')

def activation(input_, act_func):
    if act_func == 'relu':
        return np.maximum(input_, np.zeros(input_.shape))
    elif act_func == 'linear':
        return input_
    else:
        raise Exception('Activation function is not defined.')
        
  
def forward_prop(input_vec, layers_dim=layers_dim, neural_net=neural_net):
    neural_net[0].A = input_vec # Define A in input layer for for-loop convenience
    for layer_index in range(1,len(layers_dim)): # W,b,Z,A are undefined in input layer
        neural_net[layer_index].Z = np.add(np.dot(neural_net[layer_index].W, neural_net[layer_index-1].A), neural_net[layer_index].b)
        neural_net[layer_index].A = activation(neural_net[layer_index].Z, neural_net[layer_index].activation)
    return neural_net[layer_index].A  


def get_loss(y, y_hat, metric='mse'):
    if metric == 'mse':
        individual_loss = 0.5 * (y_hat - y) ** 2
        return np.mean([np.linalg.norm(individual_loss[:,col], 2) for col in range(individual_loss.shape[1])])
    else:
        raise Exception('Loss metric is not defined.')

def get_dZ_from_loss(y, y_hat, metric):
    if metric == 'mse':
        return y_hat - y
    else:
        raise Exception('Loss metric is not defined.')

def get_dactivation(A, act_func):
    if act_func == 'relu':
        return np.maximum(np.sign(A), np.zeros(A.shape)) # 1 if backward input >0, 0 otherwise; then diaganolize
    elif act_func == 'linear':
        return np.ones(A.shape)
    else:
        raise Exception('Activation function is not defined.')
        
        
def backward_prop(y, y_hat, metric='mse', layers_dim=layers_dim, neural_net=neural_net, num_train_datum=num_train_datum):
    for layer_index in range(len(layers_dim)-1,0,-1):
        if layer_index+1 == len(layers_dim): # if output layer
            dZ = get_dZ_from_loss(y, y_hat, metric)
        else: 
            dZ = np.multiply(np.dot(neural_net[layer_index+1].W.T, dZ), 
                             get_dactivation(neural_net[layer_index].A, neural_net[layer_index].activation))
        print('dz: {}'.format(dZ))
        dW = np.dot(dZ, neural_net[layer_index-1].A.T) / num_train_datum
        print('dw: {}'.format(dW))
        db = np.sum(dZ, axis=1, keepdims=True) / num_train_datum
        
        neural_net[layer_index].dW = dW
        neural_net[layer_index].db = db

learning_rate = 0.01
max_epoch = 1

for epoch in range(1,max_epoch+1):
    y_hat_train = forward_prop(X_train) # update y_hat
    print('y_hat: {}'.format(y_hat_train))
    backward_prop(y_train, y_hat_train) # update (dW,db)
    
    for layer_index in range(1,len(layers_dim)):        # update (W,b)
        neural_net[layer_index].W = neural_net[layer_index].W - learning_rate * neural_net[layer_index].dW
        neural_net[layer_index].b = neural_net[layer_index].b - learning_rate * neural_net[layer_index].db
    
    if epoch % 100000 == 0:
        print(f'{get_loss(y_train, y_hat_train):.4f}')

# test loss

print(get_loss(y_test, forward_prop(X_test)))

def predict(X_raw_any):
    X_any = np.array([standardize(X_raw_any[row,:], X_scalers[row]) for row in range(X_num_row)])
    y_hat = forward_prop(X_any)
    y_hat_any = np.array([unstandardize(y_hat[row,:], y_scalers[row]) for row in range(y_num_row)])
    return y_hat_any
    
predict(np.array([[30,70],[70,30],[3,5],[888,122]]).T)
