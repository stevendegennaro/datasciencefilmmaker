import json
from typing import List
from dsfs_vocab import Vocabulary,save_vocab, load_vocab
from dsfs_deep import (SimpleRnn,Linear,Momentum,Model,SoftMaxCrossEntropy,
                        softmax,sample_from,save_weights)
import random
import tqdm
import numpy as np
from player import Player

#Import the names from the file
namesfile="data/all_names.json"

with open(namesfile,"r") as f:
    entries = json.load(f)
players = [Player(entry) for entry in entries]
firstnames = [player.firstname for player in players if player.firstname is not None]
lastnames = [player.lastname for player in players]
suffixes = [player.suffix for player in players]

# Define the start and end characters of every name
START = "^"
STOP = "$"

def train(model: Model, 
          names: List, 
          batchsize: int, 
          n_epochs: int, 
          weightfile, 
          vocab: Vocabulary):
    for epoch in range(n_epochs):
        random.shuffle(names)
        batch = names[:batchsize]
        epoch_loss = 0
        for name in tqdm.tqdm(batch):
            model.layers[0].reset_hidden_state()
            model.layers[1].reset_hidden_state()
            name = START + name + STOP
            for prev,nexts in zip(name,name[1:]):
                inputs = vocab.one_hot_encode(prev)
                targets = vocab.one_hot_encode(nexts)
                predicted = model.forward(inputs)
                epoch_loss += model.loss.loss(predicted,targets)
                gradient = model.loss.gradient(predicted,targets)
                model.backward(gradient)
                model.optimizer.step(model)
        print(epoch,epoch_loss,generate(model, vocab))
        save_weights(model,weightfile)

def generate(model: Model, 
             vocab: Vocabulary, 
             seed_char: str = START, 
             max_len: int = 160) -> str:
    model.layers[0].reset_hidden_state()
    model.layers[1].reset_hidden_state()
    output = [seed_char]

    while output[-1] != STOP and len(output) < max_len:
        this_input = vocab.one_hot_encode(output[-1])
        predicted = model.forward(this_input)
        probabilities = softmax(predicted)
        next_char_id = sample_from(probabilities)
        output.append(vocab.get_word(next_char_id))

    return ''.join(output[1:-1])

def get_vocab(names):
    # create a Vocabulary object from a list of names
    vocab = Vocabulary([c for name in names for c in name])
    vocab.add(START)
    vocab.add(STOP)
    return vocab

def create_model(vocab, HIDDEN_DIM = 32):
    # Set up neural network
    HIDDEN_DIM = 32
    rnn1 = SimpleRnn(input_dim=vocab.size,hidden_dim=HIDDEN_DIM)
    rnn2 = SimpleRnn(input_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM)
    linear = Linear(input_dim=HIDDEN_DIM,output_dim=vocab.size)
    loss = SoftMaxCrossEntropy()
    optimizer = Momentum(learning_rate = 0.01,momentum=.9)
    model = Model([rnn1,rnn2,linear],loss,optimizer)
    return model


