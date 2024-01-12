# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import os
import sys
from datetime import datetime
import tqdm
from name_network_scratch import import_names
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Takes a list of weights and returns a random
# index chosen based on the weights
def sample_from(weights: np.array) -> int:
    total = sum(weights)
    rnd = total * np.random.random()
    for i,w in enumerate(weights):
        rnd -= w 
        if rnd <= 0: return i

def plot_history(hist_df: pd.DataFrame, color:str = "black") -> None:
    ax = fig.gca()
    ax.cla()
    ax.plot(hist_df['time'],hist_df["val_accuracy"], color=color)
    plt.draw()
    plt.pause(0.1)

class OutputHistory(keras.callbacks.Callback):
    def __init__(self, history_file):
        self.history_file = history_file
        self.hist_df = pd.DataFrame({'epoch': pd.Series(dtype='int'),
                            'time': pd.Series(dtype='float'),
                            'loss': pd.Series(dtype='float'),
                            'accuracy': pd.Series(dtype='float'),
                            'val_loss': pd.Series(dtype='float'),
                            'val_accuracy': pd.Series(dtype='float')
                           })

    def on_train_begin(self, logs={}):
        self.start_time = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        row = pd.DataFrame([logs])
        row['epoch'] = epoch
        row['time'] = elapsed_time
        self.hist_df = pd.concat([self.hist_df,row])
        with open(self.history_file, mode='w') as f:
            self.hist_df.to_csv(f, index = False)

### This is currrently very kludgey, but it's not worth 
### the time at the moment to do something more elegant
def scheduler(epoch: int, lr: float) -> float:
    if epoch < 5:
        return 0.01
    elif epoch < 50:
        return 0.001
    else:
        return 0.0005

# Use the trained network to generate a single new name
def generate(model: keras.models, vocab: Vocabulary) -> str:
    # Start with our starting character
    string = START
    x = np.zeros((1, maxlen, vocab.size))
    x[0, 0, vocab.w2i[START]] = 1.0
    # Encode the starting character
    for t in range(1,maxlen):
        start_time = timer()
        # Generate the next character
        probabilities = model.predict(x, verbose=0)[0]
        next_letter = vocab.i2w[sample_from(probabilities)]
        string += next_letter
        # If this is our STOP character, we're done
        if string[-1] == STOP:
            return string[1:-1]
        # If not, then add this to our string and
        # go back through the loop again.
        x[0, t, vocab.w2i[string[t]]] = 1.0
    # If we get here, it means we hit our max length
    # So return the string without the start character
    return string[1:]

### Shuffle names so we can split into train and test data
### I only ran once and stored these in a file so I could
### get a clean comparison between different networks using
### the same train and test data
def shuffle_names() -> list[str]:
    #Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    names = {}
    names['firstnames'] = [(START + player.firstname + STOP) \
                            for player in players if player.firstname is not None]
    names['lastnames'] = [(START + player.lastname + STOP) for player in players]
    np.random.shuffle(names['firstnames'])
    np.random.shuffle(names['lastnames'])
    with open(shufflefile,'w') as f:
        json.dump(names,f)
    return names

### Creates and trains a network on entire data set and stores in a file
    # tr = list of how many epochs to train [firstnames,lastnames]
    # gn = number of names to generate when finished ('None' if you don't want)
    # batch_size = keras batch_size, i.e. the number of names to run through
        # the network before updating. Each epoch still uses all names
    # cont = continue from the last run?
def train_network(tr: list[int,int] = [0,0], 
                  gn: int = 20,
                  learning_rate: float = 0.01, 
                  batch_size: int = None, 
                  cont: bool = False) -> None:
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    # Define the start and end characters of every name
    global START, STOP, maxlen, fig
    START = "^"
    STOP = "$"

    # Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    names = {}
    names['firstnames'] = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
    names['lastnames'] = [(START + player.lastname + STOP) for player in players]
    suffixes = [player.suffix for player in players]

    # Load the vocab file
    vocabfile = f'finalweights/vocab.txt'  # Where to find the vocab file
    if os.path.isfile(vocabfile):
        print(f"Loading vocab file ({vocabfile})")
        vocab = load_vocab(vocabfile)
    else:
        print("Vocab file does not exit")
        sys.exit()

    runs = ['firstnames','lastnames']
    generated_names = {'firstnames':[],'lastnames':[]}

    for r,run in enumerate(runs):
        model_file = f'tfweights/{run}.keras'
        history_file = f'tfweights/{run}.history'

        # Build the training set. Takes the list of names and
        # creates a list of strings and targets, e.g. "^Chuck#" becomes:
        # inputs:  ['^',    '^C',   '^Ch',  '^Chu', '^Chuc', '^Chuck']
        # targets: ['C',    'h',    'u',    'c',    'k',     '#'     ]
        print(f"Building training set for {run}")
        inputs = []
        targets = []
        for name in names[run]:
            for i in range(1,len(name)):
                inputs.append(name[:i])
                targets.append(name[i])

        # Global variable needed to define the size of the network input
        maxlen = max(len(string) for string in inputs)

        # One-hot encode inputs and targets and put into np array
        print("One-hot encoding inputs and targets")
        x = np.zeros((len(inputs), maxlen, vocab.size), dtype=np.float32)
        y = np.zeros((len(inputs), vocab.size), dtype=np.float32)
        for i, string in enumerate(inputs):
            for t, char in enumerate(string):
                x[i, t, vocab.w2i[char]] = 1
            y[i, vocab.w2i[targets[i]]] = 1

        # For the masking layer
        padding_value = np.zeros((vocab.size,))

        if cont:
            if os.path.isfile(model_file):
                model = keras.models.load_model(model_file)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            HIDDEN_DIM = 128
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

        if tr[r]:
            output_callback = OutputHistory(history_file)
            schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
            checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                                 monitor='accuracy',
                                                                 mode='max',
                                                                 save_best_only=True)
            history = model.fit(x, 
                                y, 
                                epochs=tr[r], 
                                batch_size = batch_size, 
                                callbacks = [output_callback,schedule_callback,checkpoint_callback])

        if gn:
            print(f"Generating {run}")
            for _ in range(gn):
                generated_names[run].append(generate(model,vocab))


    def random_suffix() -> str:
        suffix = np.random.choice(suffixes)
        return suffix if suffix is not None else ""

    for i in range(gn):
        print(generated_names['firstnames'][i],generated_names['lastnames'][i],random_suffix())

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total run time = ",difference)

def training_speed_test(n_epochs: int = 20, 
                        learning_rate: float = 0.01, 
                        batch_size: int = None, 
                        cont: bool = False,
                        plot_histories: bool = False) -> None:

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
            HIDDEN_DIM = 128
            model = keras.Sequential(
                [
                    keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    keras.layers.SimpleRNN(HIDDEN_DIM,),
                    keras.layers.Dense(vocab.size, activation="softmax"),
                ]
            )
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss="categorical_crossentropy", 
                          optimizer=optimizer, 
                          metrics=['accuracy'])

        start_time = datetime.now()
        print(f"Start time = ",start_time.strftime('%H:%M:%S'))

        output_callback = OutputHistory(history_file)
        schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
        history = model.fit(x, 
                            y, 
                            epochs=n_epochs, 
                            batch_size = batch_size, 
                            validation_split = 0.2,
                            callbacks = [output_callback,schedule_callback])

        model.save(model_file)

        end_time = datetime.now()
        difference = end_time - start_time
        print(f"Total run time for {run} = ", difference)

def generation_test(n_players: int) -> dict:
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    # Load in the names of the players and
    # set the filenames to be used to read/store
    # vocab info and network weights
    firstnames, lastnames, _ = import_names()
    firstnames = set([name[1:-1] for name in firstnames])
    lasstnames = set([name[1:-1] for name in lastnames])

    names = {'firstnames': firstnames,'lastnames':lastnames}
    duplicates = {}

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    for key in names:
        # Set up neural network
        model_file = f'tfweights/{key}_0.01_1000_128.keras'
        model = keras.models.load_model(model_file)
        maxlen = model.layers[0].output_shape[1]

        start_time = datetime.now()
        print(f"Generating {key} names",start_time.strftime('%H:%M:%S'))
        generated_names = []

        for _ in tqdm.tqdm(range(n_players)):
            generated_names.append(generate(model, vocab))

        end_time = datetime.now()
        difference = end_time - start_time
        print(f"Total generation time = ",difference)

        duplicates[key] = []
        count = 0
        for name in tqdm.tqdm(generated_names):
            if name in names[key]:
                count += 1
                duplicates[key].append(name)

        print(f"{count} names were already in the list ({count/n_players*100}%)")

    return duplicates

def generation_timing_test(n_players: int) -> tuple[list[float],list[float]]:
    tf.compat.v1.disable_eager_execution()
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    global START, STOP, maxlen, generation_times
    generation_times = []
 
    START = "^"
    STOP = "$"

    # Set up neural network
    model_file = f'finalweights/lastnames_rnn.keras'
    model = keras.models.load_model(model_file)
    maxlen = model.layers[0].output_shape[1]

    # times = pd.DataFrame({'length': pd.Series(dtype='int'),
    #                       'time': pd.Series(dtype='float')})

    rows = []
    for _ in tqdm.tqdm(range(n_players)):
        start_time = timer()
        name = generate(model, vocab)
        end_time = timer()
        difference = end_time - start_time
        rows.append([len(name),difference])

    times = pd.DataFrame(rows)
    times.columns = ['length','time']

    generation_times = pd.DataFrame(generation_times,columns = ['t','cp1','cp2','cp3'])

    return times, generation_times
