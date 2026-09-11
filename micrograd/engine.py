import math
class Value:

    def __init__(self,data, _children = (), _op = "") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other)-> Value: # other + self
        return self + other
    
    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')

        def _backward() ->None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other) -> Value: # other * self
        return self * other
    
    def __pow__(self,other) -> Value:
        assert isinstance(other, (int, float)), "only support integer and float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward() -> None:
            self.grad += other * self.data ** (other -1) * out.grad

        out._backward = _backward
        return out
    
    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other) -> Value:
        return self + (-other)

    def __rsub__(self, other) -> Value:
        return (-self) + other
    
    def __truediv__(self, other) -> Value:
        return self * (other ** -1)

    def __rtruediv__(self, other) -> Value:
        return (self ** -1) * other
    
    def exp(self) -> Value:
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward() -> None:
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out

    def tanh(self) -> Value: 
        x = self.data
        tanh = (math.exp(2*x) - 1) /(math.exp(2*x) + 1)
        out = Value(tanh, (self, ), 'tanh')

        def _backward() -> None:
            self.grad += (1 - out.data ** 2) * out.grad
            
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad
            
        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        x = self.data
        sigmoid = 1 / (1 + math.exp(-x))
        out = Value(sigmoid, (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self) -> None:
        visited = set()
        topo = []
        def topo_sort(node) -> None:

            if node in visited:
                return
            visited.add(node)
            for child in node._prev:

                topo_sort(child)
            topo.append(node)

        topo_sort(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()