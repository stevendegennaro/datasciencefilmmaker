# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

import os
import tensorflow as tf
from typing import Callable
from tensorflow import keras
from lite_model import LiteModel
import numpy as np
import pandas as pd
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import sys
from datetime import datetime
import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

### Import the names from file and return a dictionary
### containing lists of first names, last names, and suffixes (e.g. Jr)
### as well as a Vocabulary (loaded from a different file)
### The vocab is loaded from a file and was created using this 
### specific data set. If the network is run on a different 
### data set, there may be characters that aren't in the 
### vocabulary and it will need to be created
def import_names(shuffled: bool = False) -> tuple[str,Vocabulary]:
    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    # Import the names from the file
    if shuffled:
        with open("data/shuffled_names.json",'r') as f:
            names = json.load(f)
        names['suffixes'] = []
    else:
        namesfile = "data/all_names.json"
        with open(namesfile,"r") as f:
            entries = json.load(f)

        # print(entries)
        players = [Player(entry) for entry in entries]
        names = {}
        names['firstnames'] = [(START + player.firstname + STOP) \
                                for player in players if player.firstname is not None]
        names['lastnames'] = [(START + player.lastname + STOP) for player in players]
        names['suffixes'] = [player.suffix for player in players]

    vocab = load_vocab('finalweights/vocab.txt')

    return names, vocab

### Shuffle names so we can split into train and test data
### I only ran once and stored these in a file so I could
### get a clean comparison between different networks using
### the same train and test data
def shuffle_names() -> dict[str: list[str]]:
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

# Build the training set. Takes the list of names and
# creates a list of strings and targets, e.g. "^Chuck#" becomes:
# inputs:  ['^',    '^C',   '^Ch',  '^Chu', '^Chuc', '^Chuck']
# targets: ['C',    'h',    'u',    'c',    'k',     '#'     ]
# It then converts these into matrices of one-hot encoded vectors
def build_training_set(names: list, vocab: Vocabulary) -> tuple[np.array, np.array, np.array]:

    print(f"Building training set...")
    inputs = []
    targets = []
    for name in names:
        for i in range(1,len(name)):
            inputs.append(name[:i])
            targets.append(name[i])

    # Global variable needed to define the size of the network input
    global maxlen
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

    return x,y,padding_value

#########################
### Creates the model ###
#########################
def create_model(padding_value: np.array, 
                 maxlen: int,
                 vocab: Vocabulary,
                 HIDDEN_DIM: int,
                 learning_rate: float,
                 model_type: str = 'RNN') -> keras.models:
    if model_type == 'RNN':
        model = keras.Sequential(
            [
                keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                keras.layers.SimpleRNN(HIDDEN_DIM,),
                keras.layers.Dense(vocab.size, activation="softmax"),
            ]
        )
    elif model_type == 'LSTM':
        model = keras.Sequential(
            [
                keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                keras.layers.LSTM(HIDDEN_DIM,return_sequences=True),
                keras.layers.LSTM(HIDDEN_DIM,),
                keras.layers.Dense(vocab.size, activation="softmax"),
            ]
        )
    else:
        print(f"Invalid model type '{model_type}'. Exiting...")
        sys.exit()

    optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


#################
### Callbacks ###
#################

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

##################
### Schedulers ###
##################
### This is currrently very kludgey, but it's not worth 
### the time at the moment to do something more elegant
def step_down_1000(epoch: int, lr: float) -> float:
    if epoch < 10:
        return 0.01
    elif epoch < 25:
        return 0.005
    elif epoch < 50:
        return 0.002
    else:
        return 0.001

def step_down_LSTM(epoch: int, lr: float) -> float:
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    elif epoch < 25:
        return 0.002
    else:
        return 0.001

def flat(epoch: int, lr: float) -> float:
    # if epoch < 5:
        # return 0.001
    # else:
        return lr

################
### Generate ###
################
# Use the trained network to generate a single new name
def generate(model: keras.models, vocab: Vocabulary) -> str:
    # Start with our starting character
    string = START
    x = np.zeros((1, maxlen, vocab.size))
    x[0, 0, vocab.w2i[START]] = 1.0
    # Encode the starting character
    for t in range(1,maxlen):
        # Generate the next character

        probabilities = model.predict(x, verbose=0)[0]
        next_letter = vocab.i2w[np.random.choice(len(probabilities),p=probabilities)]
        string += next_letter
        # If this is our STOP character, we're done
        if string[-1] == STOP:
            # Return the name minus the START and STOP characters
            return string[1:-1]
        # If not, then add this to our string and
        # go back through the loop again.
        x[0, t, vocab.w2i[string[t]]] = 1.0
    # If we get here, it means we hit our max length
    # So return the string without the start character
    return string[1:]


###################
### Run Network ###
###################

### Creates and trains a network on entire data set and stores in a file
    # tr = list of how many epochs to train [firstnames,lastnames]
    # gn = number of names to generate when finished ('None' if you don't want)
    # batch_size = keras batch_size, i.e. the number of names to run through
    #   the network before updating. Each epoch still uses all names
    # cont = continue from the last run?
    # shuffled = run on the pre-shuffled list of names
def run_network(tr: list[int,int] = [0,0], 
                gn: int = 20,
                batch_size: int = None, 
                scheduler: str = 'step_down_1000',
                cont: bool = False,
                HIDDEN_DIM: int = 32,
                file_stem: str = '',
                model_type: str = 'RNN',
                validation_split: float = 0.0,
                learning_rate: float = 0.001,
                shuffled: bool = False) -> None:
 
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    # Converts the name of a funciton into the function
    scheduler = globals()[scheduler]

    # Load the vocab file
    names, vocab = import_names(shuffled)

    runs = ['firstnames','lastnames']
    generated_names = {'firstnames':[],'lastnames':[]}

    for r,run in enumerate(runs):
        model_file = f'tfweights/{run}_{file_stem}.keras'
        history_file = f'tfweights/{run}_{file_stem}.history'

        x,y,padding_value = build_training_set(names[run], vocab)

        if cont:
            if os.path.isfile(model_file):
                model = keras.models.load_model(model_file)
                tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            model = create_model(padding_value,
                                 maxlen,vocab,
                                 HIDDEN_DIM,
                                 learning_rate = learning_rate,
                                 model_type = model_type)

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
                                validation_split = validation_split,
                                callbacks = [output_callback,schedule_callback,checkpoint_callback])

            model.save(model_file)

        if gn:
            print(f"Generating {run}")
            for _ in range(gn):
                generated_names[run].append(generate(model,vocab))


    def random_suffix() -> str:
        suffix = np.random.choice(names['suffixes']) if names['suffixes'] else None
        return suffix if suffix is not None else ""

    for i in range(gn):
        print(generated_names['firstnames'][i],generated_names['lastnames'][i],random_suffix())

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total run time = ",difference)

### Generate a list of n_players, first and last names
### Does not generate suffixes because at the moment
### I can't be bothered and they are not part of my analysis
def generate_players(file_stem: str, n_players: int) -> dict[str: list[str]]:

    global maxlen
    names, vocab = import_names()

    runs = ['firstnames','lastnames']
    generated_names = {'firstnames':[],'lastnames':[]}

    for run in runs:
        print(f"Generating {run}...")
        model_file = f'finalweights/keras/{run}_{file_stem}.keras'
        model = keras.models.load_model(model_file)
        maxlen = model.layers[0].output_shape[1]
        model = LiteModel.from_keras_model(model)

        x = np.zeros((1, maxlen, vocab.size))
        x[0, 0, vocab.w2i[START]] = 1.0

        with tqdm.trange(n_players) as t:
            for _ in t:
                generated_names[run].append(generate(model,vocab))

    return generated_names




### Tests how long it takes to generate n_players first names
### Then calculates how many of those names are already
### in the first names list. Repeats for last names. Returns
### a dictionary with lists of the duplicates
def generation_test(file_stem: str, n_players: int) -> \
        tuple[dict[str: list[str]],dict[str: float],dict[str: float]]:

    global maxlen
    # tf.compat.v1.disable_eager_execution()
    # Load in the names of the players and vocab
    names, vocab = import_names()
    names['firstnames'] = set([name[1:-1] for name in names['firstnames']])
    names['lastnames'] = set([name[1:-1] for name in names['lastnames']])
    del names['suffixes']

    duplicates = {}
    percent_recreated = {}
    difference = {}

    for key in names:
        # Set up neural network
        model_file = f'finalweights/keras/{key}_{file_stem}.keras'
        model = keras.models.load_model(model_file)
        maxlen = model.layers[0].output_shape[1]
        model = LiteModel.from_keras_model(model)

        start_time = datetime.now()
        print(f"Generating {key} names",start_time.strftime('%H:%M:%S'))
        generated_names = []

        for _ in tqdm.tqdm(range(n_players)):
            generated_names.append(generate(model, vocab))

        end_time = datetime.now()
        difference[key] = end_time - start_time
        print(f"Total generation time = ",difference)

        duplicates[key] = []
        count = 0
        for name in tqdm.tqdm(generated_names):
            if name in names[key]:
                count += 1
                duplicates[key].append(name)

        percent_recreated[key] = count/n_players*100

        print(f"{count} names were already in the list ({percent_recreated[key]}%)")

    return duplicates, percent_recreated, difference

def manual_accuracy_test(model_file: str, method:str = 'argmax') -> None:

    names, vocab = import_names()
    if 'first' in model_file:
        x, y, padding_value = build_training_set(names['firstnames'], vocab)
    elif 'last' in model_file:
        x, y, padding_value = build_training_set(names['lastnames'], vocab)
        
    # Set up the model
    model = keras.models.load_model(model_file)
    # lmodel = model
    model = LiteModel.from_keras_model(model)

    count = 0
    with tqdm.trange(len(x)) as t:
        for i in t:
            probabilities = model.predict(x[[i]])[0]
            y_pred = np.zeros(vocab.size, dtype=np.float32)
            if method == 'argmax':
                y_pred[np.argmax(probabilities)] = 1.0
            elif method == 'sample_from':
                y_pred[np.random.choice(len(probabilities),p=probabilities)] = 1.0
            else:
                print("Invalid method...")
                sys.exit()
            if np.array_equal(y_pred, y[i]):
                count += 1
            if i > 0:
                t.set_description(f"{i} {count/i}")

def evaluate(model_file: str) -> None:

    # Load the model
    model = keras.models.load_model(model_file)

    names, vocab = import_names()
    if 'first' in model_file:
        x, y, padding_value = build_training_set(names['firstnames'], vocab)
    elif 'last' in model_file:
        x, y, padding_value = build_training_set(names['lastnames'], vocab)
    
    model.evaluate(x,y)


