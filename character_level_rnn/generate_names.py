from dsfs_vocab import Vocabulary, load_vocab
from dsfs_deep import Model,load_weights
from name_network import (firstnames, lastnames, suffixes, START, STOP, generate, 
                          get_vocab, create_model)
import random
import numpy as np

np.random.seed(1)
random.seed(1)

for i in range(2):
	if i == 0:
		names = firstnames
		weightfile = "weights/firstnameweights_all.txt"
		vocabfile = "weights/firstname_vocab.txt"
	else:
		names = lastnames
		weightfile = "weights/lastnameweights_all.txt"
		vocabfile = "weights/lastname_vocab.txt"

	vocab = load_vocab(vocabfile)

	# Set up neural network
	model = create_model(vocab)
	load_weights(model,weightfile)

	generated_names = []
	while len(generated_names) < 10000:
		newname = generate(model, vocab)
		#if newname not in names: 
		generated_names.append(newname)

	if i == 0: generated_first_names = generated_names[:]
	elif i == 1: generated_last_names = generated_names[:]

def random_suffix() -> str:
	suffix = random.choice(suffixes)
	return suffix if suffix is not None else ""

for i in range(len(generated_first_names)):
	print(generated_first_names[i],generated_last_names[i],random_suffix())

