from name_network import (firstnames, lastnames, START, STOP, train, generate, 
                          get_vocab, create_model)

# import json
# from pprint import pprint
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
from dsfs_deep import load_weights
import random
import numpy as np

np.random.seed(1)
random.seed(1)

names = lastnames
weightfile = "weights/weights.txt"
vocabfile = "weights/vocab.txt"

tr = True       # train first
cont = True    # continue training from previous weights
gen = True      # generate names when done training

if cont:
    vocab = load_vocab(vocabfile)
else:
    vocab = get_vocab(names)
    save_vocab(vocab, vocabfile)

model = create_model(vocab)

# If this is a continuation of a previous training session,
# load the weights from that session
if cont: load_weights(model,weightfile)

# Train the network
batchsize = 100   # number of names to use for each round of training
# batchsize = len(names)
n_epochs = 10      # number of rounds of training
if tr: train(model, names, batchsize, n_epochs, weightfile, vocab)

# Generate new names
if gen:
    generated_names = []
    for _ in range(100):
        newname = generate(model,vocab)
        # Make sure this isn't just one of the names in the training data
        if newname not in names: generated_names.append(newname)
    print(generated_names)
    print(len(generated_names))

with open('generated_test.txt',"a") as f:
    json.dump(generated_names,f)


