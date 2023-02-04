import torch


def is_correct(expected, output):
    return int(torch.argmax(expected) == torch.argmax(output))


def get_accuracy(expected, output):
    matching = torch.argmax(expected, -2) == torch.argmax(output, -2)
    return matching.float().mean()


class Derivable:
    """A base class to represent a function that is derivable.
    
    This means that it can be called normally, but that it also adds
    a `deriv` method to get its derivation from its output.
    
    e.g.
    
        activation_func = Derivable()
    
        activation = activation_func(1, 2, 3)
        deriviative = activation_func.deriv(activation)
    
    """
    def __call__(self, *args):
        pass
    
    def deriv(self, *args):
        pass
    
    
class Identity(Derivable):
    def __call__(self, x):
        return x
    
    def deriv(self, x):
        return 1
    
    @staticmethod
    def regularization(scale, weights):
        return 0
        
    
class Sigmoid(Derivable):
    fn = torch.nn.Sigmoid()
    
    def __call__(self, x):
        return self.fn(x)
    
    def deriv(self, a):
        return a * (1 - a)
    

class Tanh(Derivable):
    fn = torch.nn.Tanh()
    
    def __call__(self, x):
        return self.fn(x)
    
    def deriv(self, a):
        return 1 - a*a

    
class ReLU(Derivable):
    fn = torch.nn.ReLU()
    
    def __call__(self, x):
        return self.fn(x)
    
    def deriv(self, a):
        return a * (a > 0).float()

    
class CrossEntropy(Derivable):
    soft = torch.nn.LogSoftmax(dim=-2)
    
    def __call__(self, a, y):
        return -(y * self.soft(a)).sum()
    
    def deriv(self, a, y, nonlinear_deriv=None):
        return a - y


class L2(Derivable):
    def __call__(self, a, y):
        return torch.sum((y - a)**2, -2) / 2
        
    def deriv(self, a, y, nonlinear_deriv=None):
        return nonlinear_deriv(a) * (a - y)

    @staticmethod
    def regularization(scale, weights):
        return scale * weights
