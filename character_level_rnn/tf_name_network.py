# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

from tensorflow import keras
import numpy as np
import pandas as pd
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Takes a list of weights and returns a random
# index chosen based on the weights
def sample_from(weights):
    total = sum(weights)
    rnd = total * np.random.random()
    for i,w in enumerate(weights):
        rnd -= w 
        if rnd <= 0: return i

def plot_history(hist_df, color = "black"):
    ax = fig.gca()
    ax.cla()
    ax.plot(hist_df['time'],hist_df["val_accuracy"], color=color)
    plt.draw()
    plt.pause(0.1)

# Train the nerual network
def train(model, 
          vocab, 
          x, 
          y, 
          n_epochs, 
          batch_size, 
          validation_split, 
          checkpoint_path, 
          model_file,
          history_file,
          plot_histories: bool = False):

    hist_df = pd.DataFrame({'epoch': pd.Series(dtype='int'),
                            'time': pd.Series(dtype='float'),
                            'loss': pd.Series(dtype='float'),
                            'accuracy': pd.Series(dtype='float'),
                            'val_loss': pd.Series(dtype='float'),
                            'val_accuracy': pd.Series(dtype='float'),
                            'sample name': pd.Series(dtype='str'),
                           })

    # We train each epoch separately so that we can generate a new name 
    # after each epoch to see how well the network is doing
    start_time = datetime.now()
    for epoch in range(n_epochs):
        history = model.fit(x, 
                            y, 
                            epochs=1, 
                            batch_size = batch_size, 
                            validation_split = validation_split)
        print()
        print("Generating text after epoch: %d" % epoch)
        sample_name = generate(model, vocab)
        print(sample_name)
        # Save the model after every epoch in case we need to
        # interrupt and restart
        print("Saving model")
        model.save(model_file)

        # Add to history dataframe
        elapsed_time = (datetime.now() - start_time).total_seconds()
        row = pd.DataFrame(history.history)
        hist_df = pd.concat([hist_df,row])
        hist_df.iloc[-1, hist_df.columns.get_loc('epoch')] = epoch
        hist_df.iloc[-1, hist_df.columns.get_loc('time')] = elapsed_time
        hist_df.iloc[-1, hist_df.columns.get_loc('sample name')] = sample_name
        with open(history_file, mode='w') as f:
            hist_df.to_csv(f, index = False)
        if plot_histories: plot_history(hist_df)

    return hist_df

# Use the trained network to generate a new name
def generate(model, vocab):
    # Start with our starting character
    string = START
    # Keep going until we hit our ending character 
    # or we reach 100 characters total
    while string[-1] != STOP and len(string) < maxlen:
        # Blank vector to hold the one-hot encoded string
        x_next = np.zeros((1, maxlen, vocab.size))
        # One-hot encode each character in our string so far...
        for t, char in enumerate(string):
            x_next[0, t, vocab.w2i[char]] = 1.0 
        # predict the next letter
        probabilities = model.predict(x_next, verbose=0)[0]
        next_letter = vocab.i2w[sample_from(probabilities)]
        string += next_letter
    # return the string without the START and STOP characters
    return string[1:-1]

def split_names():
    #Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    names = {}
    names['firstnames'] = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
    names['lastnames'] = [(START + player.lastname + STOP) for player in players]
    np.random.shuffle(names['firstnames'])
    np.random.shuffle(names['lastnames'])
    with open(shufflefile,'w') as f:
        json.dump(names,f)
    return names

def train_network():
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    tr = 10              # Number of epochs to train (0 or False if you don't want to train)
    gn = 20             # Number of names to generate (0 or False if you don't want to generate names)
    cont = True         # Continue with previous weights (True or False)
    batch_size = 128
    learning_rate = 0.0001
    vocabfile = f'tfweights/vocab.txt'  # Where to save the vocab file

    # Define the start and end characters of every name
    START = "^"
    STOP = "$"

    #Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    names = {}
    names['firstnames'] = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
    names['lastnames'] = [(START + player.lastname + STOP) for player in players]
    suffixes = [player.suffix for player in players]
    if cont:
        if os.path.isfile(vocabfile):
            print(f"Loading vocab file ({vocabfile})")
            vocab = load_vocab(vocabfile)
        else:
            print("Vocab file does not exit")
    else:
        print("Generating vocab file")
        vocab = Vocabulary([c for name in names['firstnames'] for c in name] + \
                           [c for name in names['lastnames'] for c in name])
        save_vocab(vocab,vocabfile)
    print(vocab.w2i)

    runs = ['firstnames','lastnames']
    generated_names = {'firstnames':[],'lastnames':[]}
    for run in runs:
        model_file = f'tfweights/{run}.keras'

        # Build the training set
        print(f"Building training set for {run}")
        inputs = []
        targets = []
        for name in names[run]:
            for i in range(1,len(name)):
                inputs.append(name[:i])
                targets.append(name[i])

        maxlen = max(len(string) for string in inputs)

        # One-hot encode inputs and targets and put into np array
        print("One-hot encoding inputs and targets")
        x = np.zeros((len(inputs), maxlen, vocab.size), dtype=np.float32)
        y = np.zeros((len(inputs), vocab.size), dtype=np.float32)
        for i, string in enumerate(inputs):
            for t, char in enumerate(string):
                x[i, t, vocab.w2i[char]] = 1
            y[i, vocab.w2i[targets[i]]] = 1

        padding_value = np.zeros((vocab.size,))

        if cont:
            if os.path.isfile(model_file):
                model = keras.models.load_model(model_file)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            HIDDEN_DIM = 32
            model = keras.Sequential(
                [
                    keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                    keras.layers.SimpleRNN(HIDDEN_DIM,),
                    keras.layers.Dense(vocab.size, activation="softmax"),
                ]
            )
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        if tr:
            train(tr, batch_size)
        if gn:
            print(f"Generating {run}")
            for _ in range(gn):
                generated_names[run].append(generate())


    def random_suffix() -> str:
        suffix = np.random.choice(suffixes)
        return suffix if suffix is not None else ""

    for i in range(gn):
        print(generated_names['firstnames'][i],generated_names['lastnames'][i],random_suffix())

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total run time = ",difference)

def training_speed_test(n_epochs = 20, 
                        learning_rate = 0.001, 
                        batch_size = None, 
                        cont = False,
                        plot_histories = False):

    # Define the start and end characters of every name
    global START, STOP, maxlen, fig
    START = "^"
    STOP = "$"

    # Load the shuffled names
    shufflefile="data/shuffled_names.json"
    vocabfile = 'finalweights/vocab.txt'
    with open(shufflefile,'r') as f:
        names = json.load(f)
    vocab = load_vocab(vocabfile)

    runs = ['firstnames','lastnames']
    for run in runs:

        if plot_histories:
            fig, ax = plt.subplots()
            ax.set_xlabel("Time in Seconds")
            ax.set_ylabel("Accuracy")
            plt.ion()
            plt.show()

        model_file = f'tfweights/{run}_{learning_rate}_{batch_size}.keras'
        checkpoint_path = f"tfweights/{run}.ckpt"
        history_file = f"tfweights/{run}_{learning_rate}_{batch_size}.history"

        # Build the training set
        print(f"Building training set for {run}")
        inputs = []
        targets = []
        for name in names[run]:
            for i in range(1,len(name)):
                inputs.append(name[:i])
                targets.append(name[i])

        maxlen = max(len(string) for string in inputs)

        # One-hot encode inputs and targets and put into np array
        print("One-hot encoding inputs and targets")
        x = np.zeros((len(inputs), maxlen, vocab.size), dtype=np.float32)
        y = np.zeros((len(inputs), vocab.size), dtype=np.float32)
        for i, string in enumerate(inputs):
            for t, char in enumerate(string):
                x[i, t, vocab.w2i[char]] = 1
            y[i, vocab.w2i[targets[i]]] = 1

        padding_value = np.zeros((vocab.size,))

        if cont:
            if os.path.isfile(model_file):
                model = keras.models.load_model(model_file)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            HIDDEN_DIM = 32
            model = keras.Sequential(
                [
                    keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                    keras.layers.SimpleRNN(HIDDEN_DIM,),
                    keras.layers.Dense(vocab.size, activation="softmax"),
                ]
            )
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        start_time = datetime.now()
        print(f"Start time = ",start_time.strftime('%H:%M:%S'))

        hist_df = train(model = model, 
                        vocab = vocab,
                        x = x, 
                        y = y, 
                        n_epochs = n_epochs, 
                        batch_size = batch_size,
                        validation_split = 0.2,
                        checkpoint_path = checkpoint_path, 
                        model_file = model_file,
                        history_file = history_file,
                        plot_histories = plot_histories)
        end_time = datetime.now()
        difference = end_time - start_time
        print(f"Total run time for {run} = ", difference)

# def generation_test(n_players):
#     vocab_file = f"finalweights/vocab.txt"
#     vocab = load_vocab(vocab_file)
#     model = create_model(vocab)

#     # Load in the names of the players and
#     # set the filenames to be used to read/store
#     # vocab info and network weights
#     firstnames, lastnames, _ = import_names()
#     firstnames = set([name[1:-1] for name in firstnames])
#     lasstnames = set([name[1:-1] for name in lastnames])

#     names = {'firstnames': firstnames,'lastnames':lastnames}

#     for key in names:
#         weight_file = f"finalweights/{key}_weights.txt"

#         # Set up neural network
#         load_weights(model,weight_file)

#         start_time = datetime.now()
#         print(f"Generating {key} names",start_time.strftime('%H:%M:%S'))
#         generated_names = []

#         for _ in tqdm.tqdm(range(n_players)):
#             generated_names.append(generate(model, vocab))

#         end_time = datetime.now()
#         difference = end_time - start_time
#         print(f"Total generation time = ",difference)

#         count = 0
#         for name in tqdm.tqdm(generated_names):
#             if name in names[key]:
#                 count += 1

#         print(f"{count} names were already in the list ({count/n_players*100}%)")
#         # print(generated_names)

