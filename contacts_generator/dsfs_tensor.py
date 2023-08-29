from typing import List, Callable
import numpy as np
from dsfs_probability import inverse_normal_cdf
Tensor = list

def random_uniform(*dims: int) -> Tensor:
	if len(dims) == 1:
		return [np.random.random() for _ in range(dims[0])]
	else:
		return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
	if len(dims) == 1:
		return [mean + variance * inverse_normal_cdf(np.random.random()) for _ in range(dims[0])]
	else:
		return [random_normal(*dims[1:],mean=mean, variance=variance) for _ in range(dims[0])]

def random_tensor(*dims: int,init:str = 'normal',value:float = 0) -> Tensor:
	if init == 'normal':
		return random_normal(*dims)
	elif init == 'uniform':
		return random_uniform(*dims)
	elif init == 'xavier':
		variance = len(dims)/sum(dims)
		return random_normal(*dims,variance=variance)
	elif init == 'value':
		return random_value(*dims,value=value)
	else:
		raise ValueError(f"unknown init: {init}")


