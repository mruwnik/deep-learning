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
    
    def deriv(self, x):
        return x * (1 - x)


class L2(Derivable):
    def __call__(self, y, a):
        return torch.sum((y - a)**2, -2) / 2
        
    def deriv(self, y, a):
        return y - a

    @staticmethod
    def regularization(scale, weights):
        return scale * weights