# mgrad_five.py: peta verzija implementacije računskega grafa. Dodamo 
# logaritemsko in sigmoidno funkcijo.

import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.label}: {self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):  # other + self
        return self + other
    
    def __neg__(self): # - self
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def log(self):
        # ln(x)
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        return out

    def sigmoid(self):
        # σ(x) = 1 / (1 + e^(-x))
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        
        def _backward():
            # dσ/dx = σ(x) * (1 - σ(x))
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    
    def backward(self):
        # topološko uredi vozlišča v grafu in jih daj v seznam
        topo = []
        visited = set()
        def build_topo(v):
            v.grad = 0
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # za vsako od vozlišč v seznamu uporabi verižno pravilo
        # in dodaj gradient neposrednim predhodnikom
        self.grad = 1
        for v in reversed(topo):
            v._backward()