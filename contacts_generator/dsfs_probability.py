from typing import Tuple
import math
import numpy as np

SQRT_TWO_PI = math.sqrt(2*math.pi)

##### Probability distributions ######

def uniform_pdf(x: float) -> float:
	return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
	if x < 0: return 0
	elif x < 1: return x
	else: return 1

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	return (math.exp(-(x-mu)**2/2/sigma**2)/(SQRT_TWO_PI*sigma))

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	return(1+math.erf((x-mu) / math.sqrt(2)/sigma)) / 2

def inverse_normal_cdf(p: float, 
					   mu: float = 0,
					   sigma: float = 1, 
					   tolerance:float = 0.00001) -> float:
	assert 0 <= p < 1, "probability must be between 0 and 1" 
	if mu != 0 or sigma != 1:
		return mu + sigma * inverse_normal_cdf(p,tolerance=tolerance)
	low_z = -10.0
	hi_z = 10.0
	while hi_z - low_z > tolerance:
		mid_z = (low_z + hi_z) / 2
		mid_p = normal_cdf(mid_z)
		if mid_p < p:
			low_z = mid_z
		else:
			hi_z = mid_z
	return mid_z

def random_normal() -> float:
	return inverse_normal_cdf(np.random.random())


##### hypothesis testing #####

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float,float]:
	mu = p*n
	sigma = math.sqrt(p*(1-p)*n)
	return mu,sigma

normal_probability_below = normal_cdf

def normal_probability_above(lo: float,
							 mu: float = 0,
							 sigma: float = 1) -> float:
	return 1 - normal_cdf(lo,mu,sigma)

def normal_probability_between(lo: float,
							   hi: float,
							   mu: float = 0,
							   sigma: float = 1) -> float:
	return normal_cdf(hi, mu, sigma) - noraml_cdf(lo, mu,sigma)

def normal_probability_outside(lo: float,
							   hi: float,
							   mu: float = 0,
							   sigma: float = 1) -> float:
	return 1 - normal_probability_between(lo,hi,mu,sigma)

def normal_upper_bound(p:float, mu: float = 0, sigma: float = 1) -> float:
	return inverse_normal_cdf(p,mu,sigma)

def normal_lower_bound(p: float, mu: float = 0, sigma: float = 1) -> float:
	return inverse_normal_cdf(1-p,mu,sigma)

def normal_two_sided_bounds(p: float, 
							mu: float = 0, 
							sigma: float = 1) -> Tuple[float,float]:
	tail_p = (1-p)/2
	upper_bound = normal_lower_bound(tail_p, mu, sigma)
	lower_bound = normal_upper_bound(tail_p, mu, sigma)
	return lower_bound,upper_bound

def two_sided_p_value(x: float, mu: float = 0, sigma: float=1) -> float:
	if x >= mu:
		return 2 * normal_probability_above(x, mu, sigma)
	else:
		return 2 * normal_probability_below(x, mu, sigma)


