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
from timeit import default_timer as timer

def import_names(namesfile="data/all_names.json"):
    # Define the start and end characters of every name
    global START, STOP
    START = "^"
    STOP = "$"

    # Import the names from the file and put into lists
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    firstnames = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
    lastnames = [(START + player.lastname + STOP) for player in players]
    suffixes = [player.suffix for player in players]

    return firstnames, lastnames, suffixes

def calculate_accuracy(model, test_names, vocab):
    n_correct = 0
    count = 0
    for name in test_names[:100]:
        model.layers[0].reset_hidden_state()
        model.layers[1].reset_hidden_state()
        for letter,next_letter in zip(name,name[1:]):
            inputs = vocab.one_hot_encode(letter)
            targets = vocab.one_hot_encode(next_letter)
            predicted = model.forward(inputs)
            probabilities = softmax(predicted)
            next_char_predicted = vocab.i2w[sample_from(probabilities)]
            if next_char_predicted == next_letter: 
                n_correct += 1
            count += 1
    accuracy = n_correct / count
    return accuracy

def plot_history(history, color = "black"):
    ax = fig.gca()
    ax.plot(history['Time'],history["Accuracy"], color=color)
    plt.draw()
    plt.pause(0.1)

#### Training loop, called by other functions
def train(model: Model, 
          train_names: pd.DataFrame,
          test_names: pd.DataFrame, 
          batch_size: int, 
          n_epochs: int, 
          vocab: Vocabulary,
          history: pd.DataFrame,
          history_file: str,
          weight_file: str,
          plot_histories: bool = False
         ):

    start_time = datetime.now()
    print(f"Training start time = ",start_time.strftime('%H:%M:%S'))

    for epoch in range(n_epochs):
        epoch_loss = 0
        np.random.shuffle(train_names)
        batch = train_names[:batch_size]
        for name in tqdm.tqdm(batch):
            model.layers[0].reset_hidden_state()
            model.layers[1].reset_hidden_state()
            for letter,next_letter in zip(name,name[1:]):
                inputs = vocab.one_hot_encode(letter)
                targets = vocab.one_hot_encode(next_letter)
                predicted = model.forward(inputs)
                epoch_loss += model.loss.loss(predicted,targets)
                gradient = model.loss.gradient(predicted,targets)
                model.backward(gradient)
                model.optimizer.step(model)

        # Calculate the accuracy using the testing data if it exists,
        # Otherwise, calculate the accuracy using the trianing data
        accuracy = calculate_accuracy(model, test_names, vocab) \
                    if test_names \
                    else calculate_accuracy(model, train_names, vocab)

        # Output various parameters to the screen
        batch_time = datetime.now() - start_time
        print(f"Epoch: {epoch}  Epoch Loss: {epoch_loss}  Accuracy: {accuracy}  Elapsed Time: {batch_time}")
        sample_name = generate(model, vocab)
        print("Sample name: ",sample_name)
        total_time = (batch_time + elapsed_time).total_seconds()

        # Append data to the end of history DataFrame and output/plot as requested
        history.loc[len(history)] = [epoch, total_time, epoch_loss, accuracy, sample_name]
        if history_file:
            with open(history_file, 'w') as f:
                history.to_csv(f, index = False)
        if plot_histories:
            plot_history(history)
        save_weights(model,weight_file)

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total training time = ",difference)

def generate(model: Model, 
             vocab: Vocabulary, 
             max_len: int = 160) -> str:
    model.layers[0].reset_hidden_state()
    model.layers[1].reset_hidden_state()
    output = [START]

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

def create_model(vocab, HIDDEN_DIM = 32, learning_rate = 0.01, momentum = 0.9):
    # Set up neural network
    HIDDEN_DIM = 32
    rnn1 = SimpleRnn(input_dim=vocab.size,hidden_dim=HIDDEN_DIM)
    rnn2 = SimpleRnn(input_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM)
    linear = Linear(input_dim=HIDDEN_DIM,output_dim=vocab.size)
    loss = SoftMaxCrossEntropy()
    optimizer = Momentum(learning_rate = learning_rate, momentum = momentum)
    model = Model([rnn1,rnn2,linear],loss,optimizer)
    return model

#### Main method to train a network from scratch 
#### or continue from a previous training session
def train_network(which_names = 'lastnames', 
                  tr = True, 
                  cont = False, 
                  gen = True,
                  seed = None,
                  batch_size = None,
                  n_epochs = 10,
                  plot = True):

    # tr =  train the network
    # cont = continue training from previous weights
    # gen = generate names when done training

    # These are used to plot our progess during training
    global elapsed_time
    global fig

    if seed: np.random.seed(seed)

    # Load in the names of the players and
    # set the filenames to be used to read/store
    # vocab info and network weights
    firstnames, lastnames, suffixes = import_names()
    if which_names == 'lastnames':
        names = lastnames
    elif which_names == 'firstnames':
        names = firstnames
    else:
        assert False, 'Must choose firstnames or lastnames'
    weight_file = f"weights/{which_names}_weights.txt"
    history_file = f"weights/{which_names}_history.txt"

    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    # If this is a continuation of a previous training session,
    # load the vocab and weights from that session
    # If not, create new ones.
    if cont:
        model = create_model(vocab)
        load_weights(model,weight_file)
        history = pd.read_csv(history_file)
        elapsed_time = timedelta(seconds = history["Time"].iloc[-1])
    else:
        model = create_model(vocab)
        history = pd.DataFrame(columns = ["Epoch", "Time","Epoch Loss","Accuracy","Sample Name"])
        elapsed_time = timedelta(0)

    # If the user hasn't specified a batch size,
    # train on all of the names for each epoch
    if not batch_size: batch_size = len(names)

    # Plot during training
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time in Seconds")
        ax.set_ylabel("Accuracy")
        plt.ion()
        plt.show()
        plot_histories = True
    else:
        plot_histories = False

    # Train the network
    if tr: 
        train(model = model, 
              train_names = names,
              test_names = None, 
              batch_size = batch_size, 
              n_epochs = n_epochs, 
              weight_file = weight_file, 
              vocab = vocab,
              history = history,
              history_file = history_file,
              plot_histories = plot_histories)

    # Generate new names
    if gen:
        generated_names = []
        for _ in range(100):
            generated_names.append(generate(model,vocab))
        print(generated_names)

        with open('generated_test.txt',"a") as f:
            json.dump(generated_names,f)

#### Generate a batch of first and last names 
#### based on final weights from the training sessions
def generate_players(n_players, file = None):
    
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)
    model = create_model(vocab)
    for i in range(2):
        if i == 0: weight_file = "finalweights/firstnames_weights.txt"
        else: weight_file = "finalweights/firstnames_weights.txt"

        # Set up neural network
        load_weights(model,weight_file)

        print(f"Generating {'last' if i else 'first'} names")
        generated_names = []
        while len(generated_names) < n_players:
            newname = generate(model, vocab)
            # print(newname)
            generated_names.append(newname)

        if i == 0: generated_first_names = generated_names[:]
        elif i == 1: generated_last_names = generated_names[:]

    def random_suffix() -> str:
        suffix = np.random.choice(suffixes)
        return suffix if suffix is not None else ""

    for i in range(len(generated_first_names)):
        print(generated_first_names[i],generated_last_names[i],random_suffix())

def training_speed_test(n_epochs, 
                        learning_rate = 0.01, 
                        batch_size = None, 
                        cont = False,
                        momentum = 0.9):

    # These are used to plot our progess during training
    global elapsed_time
    global fig
    global START, STOP
    START = "^"
    STOP = "$"


    # Load the shuffled names
    shufflefile="data/shuffled_names.json"
    vocab_file = 'finalweights/vocab.txt'
    with open(shufflefile,'r') as f:
        names = json.load(f)
    vocab = load_vocab(vocab_file)
    print(vocab.w2i)

    runs = ['firstnames','lastnames']
    # runs = ['firstnames']
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

        model = create_model(vocab, learning_rate = learning_rate, momentum = momentum)
        weight_file = f'weights/{run}_{learning_rate}_{batch_size}_{momentum}_weights.txt'
        history_file = f'weights/{run}_{learning_rate}_{batch_size}_{momentum}_history.txt'
        weight_file = f'weights/{run}_weights.txt'
        history_file = f'weights/{run}_history.txt'
        if cont: 
            load_weights(model,weight_file)
            history = pd.read_csv(history_file)
            elapsed_time = timedelta(seconds = history["Time"].iloc[-1])
        else:
            history = pd.DataFrame(columns = ["Epoch", "Time","Epoch Loss","Accuracy","Sample Name"])
            elapsed_time = timedelta(0)
        train(model, 
              train_names = train_names,
              test_names = test_names,
              batch_size = batch_size,
              n_epochs = n_epochs,
              weight_file = weight_file, 
              vocab = vocab,
              history = history,
              history_file = history_file,
              plot_histories = False)

        plt.ioff()

def generation_test(n_players):
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)
    model = create_model(vocab)

    # Load in the names of the players and
    # set the filenames to be used to read/store
    # vocab info and network weights
    firstnames, lastnames, _ = import_names()
    firstnames = set([name[1:-1] for name in firstnames])
    lasstnames = set([name[1:-1] for name in lastnames])

    names = {'firstnames': firstnames,'lastnames':lastnames}

    for key in names:
        weight_file = f"finalweights/{key}_weights.txt"

        # Set up neural network
        load_weights(model,weight_file)

        start_time = datetime.now()
        print(f"Generating {key} names",start_time.strftime('%H:%M:%S'))
        generated_names = []

        for _ in tqdm.tqdm(range(n_players)):
            generated_names.append(generate(model, vocab))

        end_time = datetime.now()
        difference = end_time - start_time
        print(f"Total generation time = ",difference)

        count = 0
        for name in tqdm.tqdm(generated_names):
            if name in names[key]:
                count += 1

        print(f"{count} names were already in the list ({count/n_players*100}%)")
        # print(generated_names)
        # print(names[key])

    # for i in range(len(generated_first_names)):
    #     print(generated_first_names[i],generated_last_names[i],random_suffix())

def generation_timing_test(n_players):
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    # Set up neural network
    weight_file = f"finalweights/lastnames_weights.txt"
    model = create_model(vocab)
    load_weights(model,weight_file)

    rows = []
    for _ in tqdm.tqdm(range(n_players)):
        start_time = timer()
        name = generate(model, vocab)
        end_time = timer()
        difference = end_time - start_time
        rows.append([len(name),difference])

    times = pd.DataFrame(rows)
    times.columns = ['length','time']

    return times


