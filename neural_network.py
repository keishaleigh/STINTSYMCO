import torch
import torch.nn as nn
import torch.nn.init as init

""" modified version of nn file from notebook 5
    key changes:
    - customize initialization parameters
    - regularization via dropout (better for our dataset size) (i think)
    - use batch normalization for more stable training
    - choose between different activation functions
    - control randomness via seed

    didnt change
"""

class NeuralNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 list_hidden,
                 activation='relu',
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 weight_init_mean=0.0,
                 weight_init_std=0.1,
                 bias_init=0.0,
                 seed=None):
        """Class constructor for NeuralNetwork

        Arguments:
            input_size {int} -- Number of features in the dataset
            num_classes {int} -- Number of classes in the dataset
            list_hidden {list} -- List of integers representing the number of
            units per hidden layer in the network
            activation {str, optional} -- Type of activation function. Choices
                include 'sigmoid', 'tanh', 'relu', 'leaky_relu'.
            use_batch_norm {bool, optional} -- Whether to use batch normalization
            dropout_rate {float, optional} -- Dropout rate (0 = no dropout)
            weight_init_mean {float, optional} -- Mean for weight initialization
            weight_init_std {float, optional} -- Std dev for weight initialization
            bias_init {float, optional} -- Value for bias initialization
            seed {int, optional} -- Random seed (None for random)
        """
        super(NeuralNetwork, self).__init__()
        
        if seed is not None:
            torch.manual_seed(seed)

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.bias_init = bias_init
        
        self.create_network()
        self.init_weights()

    def create_network(self):
        """Creates the layers of the neural network.
        """
        layers = []
        in_features = self.input_size

        # Create hidden layers
        for i, out_features in enumerate(self.list_hidden):
            layers.append(nn.Linear(in_features, out_features))
            
            # Add batch normalization if enabled
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            
            # Add activation
            layers.append(self.get_activation())
            
            # Add dropout if enabled
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, self.num_classes))
        layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        """Initializes the weights of the network.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, 
                           mean=self.weight_init_mean, 
                           std=self.weight_init_std)
                init.constant_(module.bias, self.bias_init)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def get_activation(self):
        """Returns the activation function layer.
        """
        if self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:  # Default to ReLU
            return nn.ReLU(inplace=True)

    def forward(self, x, verbose=False):
        """Forward propagation of the model.
        """
        if verbose:
            print("Input shape:", x.shape)
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if verbose:
                print(f'Layer {i} ({layer.__class__.__name__}) output shape:', x.shape)
        
        probabilities = self.layers[-1](x)
        
        if verbose:
            print('Softmax output shape:', probabilities.shape)
        
        return x, probabilities

    def predict(self, probabilities):
        """Returns the index of the class with the highest probability.
        """
        return torch.argmax(probabilities, dim=1)