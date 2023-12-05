import json
from typing import List
from dsfs_vocab import Vocabulary,save_vocab, load_vocab
from dsfs_deep import (SimpleRnn,Linear,Momentum,Model,SoftMaxCrossEntropy,
                        softmax,sample_from,save_weights,load_weights)
import tqdm
import numpy as np
from player import Player
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
import pandas as pd

# np.random.seed(1)

# Define the start and end characters of every name
START = "^"
STOP = "$"

#Import the names from the file
namesfile="data/all_names.json"
with open(namesfile,"r") as f:
    entries = json.load(f)
players = [Player(entry) for entry in entries]
firstnames = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
lastnames = [(START + player.lastname + STOP) for player in players]
suffixes = [player.suffix for player in players]

def calculate_accuracy(model, test_names, vocab):
    n_correct = 0
    count = 0
    for name in test_names[:100]:
        model.layers[0].reset_hidden_state()
        model.layers[1].reset_hidden_state()
        for prev,nexts in zip(name,name[1:]):
            inputs = vocab.one_hot_encode(prev)
            targets = vocab.one_hot_encode(nexts)
            predicted = model.forward(inputs)
            probabilities = softmax(predicted)
            next_char_predicted = vocab.i2w[sample_from(probabilities)]
            if next_char_predicted == nexts: 
                n_correct += 1
            count += 1
    accuracy = n_correct / count
    return accuracy

def plot_loss(losses, color = "black"):
    ax = fig.gca()
    ax.plot(losses['Time'],losses["Accuracy"], color=color)
    plt.draw()
    plt.pause(0.1)

def train(model: Model, 
          train_names: List,
          test_names: List, 
          batch_size: int, 
          n_epochs: int, 
          weightfile, 
          vocab: Vocabulary,
          losses,
          lossesfile,
          plot_losses = False):

    start_time = datetime.now()
    print(f"Training start time = ",start_time.strftime('%H:%M:%S'))

    for epoch in range(n_epochs):
        epoch_loss = 0
        np.random.shuffle(train_names)
        batch = train_names[:batch_size]
        for name in tqdm.tqdm(batch):
        # for name in batch:
            model.layers[0].reset_hidden_state()
            model.layers[1].reset_hidden_state()
            for prev,nexts in zip(name,name[1:]):
                inputs = vocab.one_hot_encode(prev)
                targets = vocab.one_hot_encode(nexts)
                predicted = model.forward(inputs)
                epoch_loss += model.loss.loss(predicted,targets)
                gradient = model.loss.gradient(predicted,targets)
                model.backward(gradient)
                model.optimizer.step(model)
        accuracy = calculate_accuracy(model, test_names, vocab) \
                    if len(test_names) \
                    else calculate_accuracy(model, train_names, vocab)
        batch_time = datetime.now() - start_time
        print(f"Epoch: {epoch}  Epoch Loss: {epoch_loss}  Accuracy: {accuracy}  Elapsed Time: {batch_time}")
        sample_name = generate(model, vocab)
        print("Sample name: ",sample_name)
        total_time = (batch_time + elapsed_time).total_seconds()
        losses.loc[len(losses)] = [epoch,total_time, epoch_loss, accuracy, sample_name]
        save_weights(model,weightfile)
        if lossesfile:
            with open(lossesfile, 'w') as f:
                losses.to_csv(f, index = False)
        if plot_losses:
            plot_loss(losses)

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total training time = ",difference)

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
    return vocab

def create_model(vocab, HIDDEN_DIM = 32, learning_rate = 0.01):
    # Set up neural network
    HIDDEN_DIM = 32
    rnn1 = SimpleRnn(input_dim=vocab.size,hidden_dim=HIDDEN_DIM)
    rnn2 = SimpleRnn(input_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM)
    linear = Linear(input_dim=HIDDEN_DIM,output_dim=vocab.size)
    loss = SoftMaxCrossEntropy()
    optimizer = Momentum(learning_rate = learning_rate, momentum=.9)
    model = Model([rnn1,rnn2,linear],loss,optimizer)
    return model

def train_network_full():
    np.random.seed(1)
    names = lastnames
    weightfile = "weights/weights.txt"
    vocabfile = "weights/vocab.txt"

    tr = True       # train first
    cont = False    # continue training from previous weights
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
    batch_size = 100   # number of names to use for each round of training
    # batch_size = len(names)
    n_epochs = 10      # number of rounds of training
    if tr: train(model, names, [], batch_size, n_epochs, weightfile, vocab, None)

    # Generate new names
    if gen:
        generated_names = []
        for _ in range(100):
            generated_names.append(generate(model,vocab))
        print(generated_names)

    with open('generated_test.txt',"a") as f:
        json.dump(generated_names,f)

def generate_players():
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

        print("Generating names")
        generated_names = []
        while len(generated_names) < 10000:
            newname = generate(model, vocab)
            #if newname not in names: 
            generated_names.append(newname)

        if i == 0: generated_first_names = generated_names[:]
        elif i == 1: generated_last_names = generated_names[:]

    def random_suffix() -> str:
        suffix = np.random.choice(suffixes)
        return suffix if suffix is not None else ""

    for i in range(len(generated_first_names)):
        print(generated_first_names[i],generated_last_names[i],random_suffix())

def analyze_network(n_epochs, 
                     learning_rate = 0.01, 
                     batch_size = None, 
                     cont = False):

    global elapsed_time
    global fig

    # Load the shuffled names
    shufflefile="data/shuffled_names.json"
    vocabfile = 'tfweights/vocab_analyze.txt'
    with open(shufflefile,'r') as f:
        names = json.load(f)
    vocab = load_vocab(vocabfile)
    print(vocab.w2i)

    runs = ['firstnames','lastnames']
    runs = ['firstnames']
    runs = ['lastnames']

    for run in runs:

        n_train = int(np.floor(len(names[run]) * 0.8))
        train_names = names[run][:n_train]
        test_names = names[run][n_train:]
        if not batch_size:
            batch_size = len(train_names)

        fig, ax = plt.subplots()
        ax.set_xlabel("Time in Seconds")
        ax.set_ylabel("Accuracy")
        title = f"Scratch Network Trained on "
        title += f"{'First Names' if run == 'firstnames' else 'Last Names'}\n"
        title += f"Batch Size = {batch_size}, Learning Rate = {learning_rate}"
        ax.set_title(title)
        plt.ion()
        plt.show()

        model = create_model(vocab, learning_rate = learning_rate)
        weightfile = f'weights/{run}_{learning_rate}_{batch_size}_weights.txt'
        lossesfile = f'weights/{run}_{learning_rate}_{batch_size}_history.txt'
        if cont: 
            load_weights(model,weightfile)
            losses = pd.read_csv(lossesfile)
            elapsed_time = timedelta(seconds = losses["Time"].iloc[-1])
        else:
            losses = pd.DataFrame(columns = ["Epoch", "Time","Epoch Loss","Accuracy","Sample Name"])
            elapsed_time = timedelta(0)
        train(model, 
              train_names = train_names,
              test_names = test_names,
              batch_size = batch_size,
              n_epochs = n_epochs,
              weightfile = weightfile, 
              vocab = vocab,
              losses = losses,
              lossesfile = lossesfile,
              plot_losses = True)

        plt.ioff()



