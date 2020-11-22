import torch
from torch import Tensor
import torch.nn as nn
from torch import matmul
from torch import tanh
from torch import Tensor
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch import sigmoid
torch.set_default_dtype(torch.float64)

# Regression model: straight-up neural network for eg., regressing the mean and logvar separately
class RegressionNetwork(nn.Module):
    
    # init
    def __init__(self, n_turbine_features, n_weather_features, n_units=[64,64], activation=leaky_relu, init=0):
        
        super(RegressionNetwork, self).__init__()
        
        # architecture: (n_turbines, n_weather) -> hidden layers (n_units) -> 2 outputs (mean and log variance)
        self.n_turbine_features = n_turbine_features
        self.n_weather_features = n_weather_features
        self.n_units = n_units
        
        # network weights
        self.W1t = nn.Parameter(torch.Tensor(self.n_turbine_features, self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W1w = nn.Parameter(torch.Tensor(self.n_weather_features, self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(self.n_units[0], self.n_units[1]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W3 = nn.Parameter(torch.Tensor(self.n_units[1], 1).normal_(mean=0, std=1e-5), requires_grad=True)
        
        # network biases
        self.b1 = nn.Parameter(torch.Tensor(self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.b2 = nn.Parameter(torch.Tensor(self.n_units[1]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.b3 = nn.Parameter(torch.Tensor(1).normal_(mean=0, std=1e-5), requires_grad=True)
        
        # initialize the bias of the output layer
        self.b3.data.fill_(init)

        # activation
        self.activation = activation
    
    # forward pass through the network
    def forward(self, turbine_features, weather_features):
        
        # forward pass through two hidden layers (including split input layer)
        output = torch.matmul(self.activation(torch.matmul(self.activation(torch.matmul(turbine_features, self.W1t) + torch.matmul(weather_features, self.W1w) + self.b1), self.W2) + self.b2), self.W3) + self.b3
               
        return output

# Gaussian Neural Density Estimator (2-layer version)
# takes some inputs (turbine properties and weather forecasts) and outputs a mean and (log) variance describing a Gaussian distribution for the predicted power output
class GaussianNDE(nn.Module):
    
    # init
    def __init__(self, n_turbine_features, n_weather_features, n_units=[64,64], activation=leaky_relu, mean_init=0, logvar_init=1.):
        
        super(GaussianNDE, self).__init__()
        
        # architecture: (n_turbines, n_weather) -> hidden layers (n_units) -> 2 outputs (mean and log variance)
        self.n_turbine_features = n_turbine_features
        self.n_weather_features = n_weather_features
        self.n_units = n_units
        
        # network weights
        self.W1t = nn.Parameter(torch.Tensor(self.n_turbine_features, self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W1w = nn.Parameter(torch.Tensor(self.n_weather_features, self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(self.n_units[0], self.n_units[1]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.W3 = nn.Parameter(torch.Tensor(self.n_units[1], 2).normal_(mean=0, std=1e-5), requires_grad=True)
        
        # network biases
        self.b1 = nn.Parameter(torch.Tensor(self.n_units[0]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.b2 = nn.Parameter(torch.Tensor(self.n_units[1]).normal_(mean=0, std=1e-5), requires_grad=True)
        self.b3 = nn.Parameter(torch.Tensor(2).normal_(mean=0, std=1e-5), requires_grad=True)
        
        # initialize the bias of the output layer
        self.b3[0].data.fill_(mean_init)
        self.b3[1].data.fill_(logvar_init)

        # activation
        self.activation = activation
    
    # forward pass through the network
    def forward(self, turbine_features, weather_features):
        
        # forward pass through two hidden layers (including split input layer)
        output = torch.matmul(self.activation(torch.matmul(self.activation(torch.matmul(turbine_features, self.W1t) + torch.matmul(weather_features, self.W1w) + self.b1), self.W2) + self.b2), self.W3) + self.b3
        
        # split the output layer into two outputs: mean nd log variance (log_variance)
        mean, log_variance = torch.split(output, 1, dim=-1)
               
        return mean, log_variance

# same as above but with generic network depth
class GaussianNDE_Deep(nn.Module):
    
    # init
    def __init__(self, n_inputs, n_units=[64,64], activation=leaky_relu):
        
        super(GaussianNDE, self).__init__()
        
        # architecture: n_inputs -> hidden layers (n_units) -> 2 outputs (mean and variance)
        self.n_inputs = n_inputs # n_inputs = number of turbine features + number of weather features (ie., total length of input feature vector)
        self.architecture = [self.n_inputs] + n_units + [2]
        self.n_layers = len(self.architecture) - 1
        
        # network weights and biases
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(self.architecture[i], self.architecture[i+1]), requires_grad=True) for i in range(self.n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.Tensor(self.architecture[i+1]), requires_grad=True) for i in range(self.n_layers)])
        
        # activation
        self.activation = activation
    
    # forward pass through the network
    def forward(self, turbine_features, weather_features):
        
        # initialize input layer: concatenate the turbine features and weather features together into a single input vector
        layers = [torch.cat((turbine_features, weather_features), -1)]
        
        # implement network
        for i in range(self.n_layers):
            layers.append(self.activation(matmul(layers[-1], self.W[i]) + self.b[i]))
        
        # split the output layer into two outputs: mean nd log variance (log_variance)
        mean, log_variance = torch.split(layers[-1], 1, dim=-1)
               
        return mean, log_variance