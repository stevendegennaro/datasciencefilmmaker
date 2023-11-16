import json
from pprint import pprint
from dsfs_vocab import Vocabulary,save_vocab, load_vocab
from dsfs_deep import (SimpleRnn,Linear,Momentum,Model,SoftMaxCrossEntropy,
                        softmax,sample_from,GradientDescent,save_weights,load_weights)
import random
import tqdm
import numpy as np
from collections import Counter
import sys

# np.random.seed(1)
# random.seed(1)

def get_company_names():
    exchanges = ['nyse','nasdaq']
    names = []
    for exchange in exchanges:
        filename = 'data/' + exchange +'.json'
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
        names.extend([company['name'] for company in data])
        names = sorted(list(set(names)))
        ##### Clean the names to remove ones that have weird punctuation ######
    return names

fullnames = get_company_names()

suffixes = []
for name in fullnames:
    suffixes.append(name.split()[-1])

suffix_counts = Counter(suffixes)

# remove common suffixes
common_suffixes = {}
for key in suffix_counts:
    if suffix_counts[key] > 12:
        common_suffixes[key] = suffix_counts[key]

names = [name.split() for name in fullnames]
for i in range(len(names)):
    if names[i][-1] in common_suffixes.keys():
        names[i] = names[i][:-1]

# Define the start and end characters of every name
START = "START_NAME"
STOP = "END_NAME"

def train(batchsize: int, n_epochs: int):
    for epoch in range(n_epochs):
        random.shuffle(names)
        batch = names[:batchsize]
        epoch_loss = 0
        for name in tqdm.tqdm(batch):
            rnn1.reset_hidden_state()
            rnn2.reset_hidden_state()
            name = [START] + name + [STOP]
            for prev,nexts in zip(name,name[1:]):
                # print(prev, nexts)
                inputs = vocab.one_hot_encode(prev)
                target = vocab.one_hot_encode(nexts)
                predicted = model.forward(inputs)
                epoch_loss += model.loss.loss(predicted,target)
                gradient = model.loss.gradient(predicted,target)
                model.backward(gradient)
                model.optimizer.step(model)
        print(epoch,epoch_loss,generate(vocab))
    save_weights(model,weightfile)

def generate(vocab: Vocabulary, seed_string: str = START, max_len: int = 160) -> str:
    rnn1.reset_hidden_state()
    rnn2.reset_hidden_state()
    output = [seed_string]

    while output[-1] != STOP and len(output) < max_len:
        this_input = vocab.one_hot_encode(output[-1])
        predicted = model.forward(this_input)
        probabilities = softmax(predicted)
        next_string_id = sample_from(probabilities)
        output.append(vocab.get_word(next_string_id))

    return ' '.join(output[1:-1])

tr = True    # train first
cont = False    # continue training from previous weights
gen = True      # generate names when done training
v = True


weightfile = "weights/weights_v2.txt"
vocabfile = "weights/vocab_v2.txt"

if v:
    vocab = load_vocab(vocabfile)
else:
# create the vocabulary object
    vocab = Vocabulary([word for name in names for word in name])
    vocab.add(START)
    vocab.add(STOP)
    save_vocab(vocab, vocabfile)

# Set up neural network
HIDDEN_DIM = 31
rnn1 = SimpleRnn(input_dim=vocab.size,hidden_dim=HIDDEN_DIM)
rnn2 = SimpleRnn(input_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM)
linear = Linear(input_dim=HIDDEN_DIM,output_dim=vocab.size)
loss = SoftMaxCrossEntropy()
optimizer = Momentum(learning_rate = 0.01,momentum=.9)
model = Model([rnn1,rnn2,linear],loss,optimizer,[],[])

# If this is a continuation of a previous training session,
# load the weights from that session
if cont: load_weights(model,weightfile)

# Train the network
batchsize = 100   # number of names to use for each round of training
# batchsize = len(names)
n_epochs = 1      # number of rounds of training
if tr: train(batchsize, n_epochs)

# Generate new names
if gen:
    generated_names = []
    for _ in range(100):
        newname = generate(vocab)
        # Make sure this isn't just one of the names in the training data
        if newname not in names: generated_names.append(newname)
    pprint(generated_names)
    print(len(generated_names))

with open('weights/generated_v2.txt',"a") as f:
    json.dump(generated_names,f)