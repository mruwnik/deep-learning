import torch
from torch import tensor

from math_funcs import L2, Sigmoid


class Network:
    def __init__(self, layers, nonlinearity=Sigmoid(), cost_func=L2()):
        # This assumes that each layer will use the same nonlinearity. Seems
        # like a good enough heuristic for now
        self.nonlinearity = nonlinearity
        self.cost_func = cost_func
        self.w = [torch.rand(n_out, n_in) / torch.sqrt(tensor(n_in)) for n_in, n_out in zip(layers, layers[1:])]
        self.b = [torch.rand(n_out, 1) for n_out in layers[1:]]
        
    def forward(self, inputs):
        """Do a forward pass and return all the activations for each layer.
        
        The inputs will be the first element of the activations.
        The actual output of the network is the last element of the resulting list.
        """
        a = inputs
        activations = [a]
        for w, b in zip(self.w, self.b):
            a = self.nonlinearity(w @ a + b)
            activations.append(a)
        return activations
         
    def step(self, inputs):
        """"Get the outputs from running the given `inputs` through the network."""
        return self.forward(inputs)[-1]
    
    def gradient(self, activations, expected):
        """Calculate the gradient(s).
        
        :param list activations: a list of per layer activations from the forward pass
        :param tensor expected: the expected output of the network, i.e. the label for
                                the inputs
        :returns: a list of per layer (dw, db) gradients, from the first layer to the last
        
        Seeing as it uses pytorch, either single items, or tensors of items can be
        provided.
        
        This is based on http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation
        and https://ahiru.pl/notes/backpropagation/ 
        """
        layer_gradients = []
       
        # Calculate the gradients for the final layer, which is
        # different from the rest in that it's based directly on
        # The cost function, as opposed to the error of a later
        # layer
        in_ = activations[-2]
        out = activations[-1]

        nabla_C = self.cost_func.deriv(out, expected)
        sigma_L = self.nonlinearity.deriv(out)
        delta_L = nabla_C * self.nonlinearity.deriv(out)

        prev_delta = delta_L
        
        dw = prev_delta @ torch.transpose(in_, -2, -1) 
        db = prev_delta 
        layer_gradients.append([dw.mean(0), db.mean(0)])
        
        # Now for each layer, starting from the last-but-one and going
        # to the first one, calculate the gradients of the weights and
        # biases
        for l in range(len(self.w) - 2, -1, -1):
            in_ = activations[l]
            out = activations[l+1]
            
            delta_l = (torch.transpose(self.w[l+1], -2, -1) @ prev_delta) * self.nonlinearity.deriv(out)
            prev_delta = delta_l
            
            dw = delta_l @ torch.transpose(in_, -2, -1)
            db = delta_l
            
            layer_gradients.append([dw.mean(0), db.mean(0)])
        
        # The `layer_gradients` list is from the last to first layer,
        # so reverse it
        return list(reversed(layer_gradients))
    
    def update(self, gradient, lr):
        # For each layer of the gradient list, update the
        # weights and biases for that layer
        for i, (w, b) in enumerate(gradient):
            self.w[i] -= lr * w
            self.b[i] -= lr * b
            
    def cost(self, expected, output):
        """Calculate the cost between the label and what the network predicted."""
        return self.cost_func(expected, output)