import time
from numpy import random

filename = "scroll.txt"
with open(filename, "r") as f:
	line = f.readline()
	while line:
		print(line)
		line = f.readline()
		time.sleep(max(random.normal(0.5,.1),0))