from typing import Iterable, Tuple, List
import numpy as np
import operator
import json

def random_tensor(*dims: int, init:str = 'normal', value:float = 0):
    if init == 'normal':
        return np.random.normal(size=dims)
    elif init == 'uniform':
        return np.random.uniform(size=dims)
    elif init == 'xavier':
        variance = len(dims)/sum(dims)
        return np.random.normal(scale=variance,size=dims)
    else:
        raise ValueError(f"unknown init: {init}")

##### Layers #####
class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def params(self) -> List[np.array]:
        return ()
    def grads(self) -> List[np.array]:
        return ()

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.array(random_tensor(output_dim,input_dim,init=init))
        self.b = np.array(random_tensor(output_dim,init=init))

    def forward(self, inputs:np.array) -> np.array:
        self.inputs = np.array(inputs)
        return np.dot(self.inputs,self.w.transpose()) + self.b

    def backward(self, grad: np.array) -> np.array:
        self.b_grad = np.array(grad)
        self.w_grad = np.outer(np.array(grad),self.inputs)
        return np.array([np.sum(np.dot(self.w.transpose()[i],self.b_grad)) for i in range(self.input_dim)])

    def params(self) -> List[np.array]:
        return [self.w, self.b]

    def grads(self) -> List[np.array]:
        return [self.w_grad, self.b_grad]

class SimpleRnn(Layer):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w = np.array(random_tensor(hidden_dim,input_dim, init='xavier'))
        self.u = np.array(random_tensor(hidden_dim,hidden_dim, init='xavier'))
        self.b = np.array(random_tensor(hidden_dim))

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden = np.zeros(self.hidden_dim)

    def forward(self, inputs:np.array) -> np.array:
        self.inputs = inputs
        self.prev_hidden = self.hidden
        self.hidden = np.tanh(np.dot(self.w,self.inputs) + np.dot(self.u,self.hidden) + self.b)
        return self.hidden

    def backward(self, grad: np.array) -> np.array:
        self.b_grad = grad * (1 - self.hidden ** 2)
        self.w_grad = np.outer(self.b_grad,self.inputs)
        self.u_grad = np.outer(self.b_grad,self.prev_hidden)
        return np.array([np.sum(np.dot(self.w.transpose()[i],self.b_grad)) for i in range(self.input_dim)])

    def params(self) -> List[np.array]:
        return [self.w, self.u, self.b]

    def grads(self) -> List[np.array]:
        return [self.w_grad, self.u_grad,self.b_grad]


##### Loss Functions #####
class Loss:
    def loss(self, predicted, actual) -> float:
        raise NotImplementedError

    def gradient(self,predicted, actual) -> np.array:
        raise NotImplementedError


def softmax(x: np.array) -> np.array:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class SoftMaxCrossEntropy(Loss):
    def loss(self,predicted:np.array,actual:np.array) -> float:
        return -np.sum(np.log(softmax(predicted) + 1e-30) * actual)

    def gradient(self,predicted:np.array,actual:np.array) -> np.array:
        return  softmax(predicted) - actual

##### Optimizers #####
class Optimizer:
    def step(self, layer:Layer) -> None:
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(),layer.grads()):
            param[:] = param - grad * self.lr

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates  = []

    def step(self,layer: Layer) -> None:
        if not self.updates:
            self.updates = [np.zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates,layer.params(),layer.grads()):
            update[:] = self.mo * update + (1 - self.mo) * grad
            param[:] = param - update * self.lr


class Model(Layer):
    def __init__(self, 
                    layers:List[Layer], 
                    loss:Loss, 
                    optimizer: Optimizer, 
                ) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.array]:
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> List[np.array]:
        return (grad for layer in self.layers for grad in layer.grads())


def save_weights(model: Layer, filename: str) -> None:
    weights = [param.tolist() for param in list(model.params())]
    with open(filename,"w") as f:
        json.dump(weights, f)

def load_weights(model: Layer, filename: str) -> None:
    with open(filename) as f:
        weights = json.load(f)

    assert all(np.shape(param) == np.shape(weight)
        for param, weight in zip(model.params(),weights))

    for param, weight in zip(model.params(),weights):
        param[:] = weight

def sample_from(weights: List[float]) -> int:
    total = sum(weights)
    rnd = total * np.random.random()
    for i,w in enumerate(weights):
        rnd -= w 
        if rnd <= 0: return i