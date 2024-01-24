# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from name_network_scratch import import_names
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def import_names(which):
    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    #Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    if which == 'first':
        names = [(START + player.firstname + STOP) \
                    for player in players if player.firstname is not None]
    elif which == 'last':
        names = [(START + player.lastname + STOP) \
            for player in players if player.lastname is not None]
    else:
        print("Invalid data set")
        sys.exit()

    return names

def build_training_set(names, vocab):

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

    return x,y


# Takes a list of weights and returns a random
# index chosen based on the weights
def sample_from(weights: np.array) -> int:
    total = sum(weights)
    rnd = total * np.random.random()
    for i,w in enumerate(weights):
        rnd -= w 
        if rnd <= 0: return i


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
def step_down_1000(epoch: int, lr: float) -> float:
    if epoch < 10:
        return 0.01
    elif epoch < 25:
        return 0.005
    elif epoch < 50:
        return 0.002
    else:
        return 0.001

def exponential(epoch: int, lr: float) -> float:
    learning_rate = initial_learning_rate * decay_rate^(step / decay_steps)


def flat(epoch: int, lr: float) -> float:
    # if epoch < 5:
        return 0.001
    # else:
        return lr

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
        next_letter = vocab.i2w[sample_from(probabilities)]
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
def run_network(tr: list[int,int] = [0,0], 
                  gn: int = 20,
                  batch_size: int = None, 
                  scheduler: str = 'flat',
                  cont: bool = False,
                  HIDDEN_DIM: int = 32,
                  file_suffix = '') -> None:
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    # Define the start and end characters of every name
    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    # Converts the name of a funciton into the function
    scheduler = globals()[scheduler]

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
            # HIDDEN_DIM = 128
            model = keras.Sequential(
                [
                    keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                    keras.layers.SimpleRNN(HIDDEN_DIM,),
                    keras.layers.Dense(vocab.size, activation="softmax"),
                ]
            )
            optimizer = keras.optimizers.RMSprop()
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        if tr[r]:
            output_callback = OutputHistory(history_file)
            schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
            checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                                 monitor='loss',
                                                                 mode='min',
                                                                 save_best_only=True)
            history = model.fit(x, 
                                y, 
                                epochs=tr[r], 
                                batch_size = batch_size, 
                                callbacks = [output_callback,schedule_callback,checkpoint_callback])

            model.save(model_file)

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
                        batch_size: int = None, 
                        cont: bool = False,
                        scheduler: str = 'step_down',
                        learning_rate = 0.01,
                        HIDDEN_DIM: int = 32,
                        file_suffix = '') -> None:

    # Define the start and end characters of every name
    global START, STOP, maxlen, fig
    START = "^"
    STOP = "$"
    scheduler = globals()[scheduler]

    # Load the shuffled names
    shufflefile="data/shuffled_names.json"
    vocabfile = 'finalweights/vocab.txt'
    with open(shufflefile,'r') as f:
        names = json.load(f)
    vocab = load_vocab(vocabfile)

    runs = ['firstnames','lastnames']
    for run in runs:

        model_file = f'tfweights/{run}_{file_suffix}.keras'
        history_file = f"tfweights/{run}_{file_suffix}.history"

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
            # HIDDEN_DIM = 128
            model = keras.Sequential(
                [
                    keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
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


### Tests how long it takes to generate n_players names
### Then calculates how many of those names are already
### in the names list. Repeats for last names. Returns a
### dictionary with lists of the duplicates
def generation_test(n_players: int) -> dict:
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    # Load in the names of the players and
    # set the filenames to be used to read/store
    # vocab info and network weights
    firstnames, lastnames, _ = import_names()
    firstnames = set([name[1:-1] for name in firstnames])
    lastnames = set([name[1:-1] for name in lastnames])

    names = {'firstnames': firstnames,'lastnames':lastnames}
    duplicates = {}

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    for key in names:
        # Set up neural network
        model_file = f'finalweights/firstnames_RNN_1000_32.keras'
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

def generation_timing_test(n_players: int) -> list[float]:
    # tf.compat.v1.disable_eager_execution()
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    # Set up neural network
    model_file = f'finalweights/lastnames_rnn.keras'
    model = keras.models.load_model(model_file)
    maxlen = model.layers[0].output_shape[1]

    times = []
    for _ in tqdm.tqdm(range(n_players)):
        start_time = timer()
        name = generate(model, vocab)
        end_time = timer()
        difference = end_time - start_time
        times.append(difference)

    return times


def manual_accuracy_test(model_file, method = 'argmax'):

    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)
    if 'first' in model_file:
        names = import_names('first')
    if 'last' in model_file:
        names = import_names('last')
    x,y = build_training_set(names, vocab)
    padding_value = np.zeros((vocab.size,))
        
    # Set up the model
    tf.get_logger().setLevel('ERROR')
    model = keras.models.load_model(model_file)
    lmodel = model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    lmodel = LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    count = 0
    with tqdm.trange(len(x)) as t:
        for i in t:
            probabilities = lmodel.predict(x[[i]])[0]
            y_pred = np.zeros(vocab.size, dtype=np.float32)
            if method == 'argmax':
                y_pred[np.argmax(probabilities)] = 1.0
            elif method == 'sample_from':
                y_pred[sample_from(probabilities)] = 1.0
            else:
                print("Invalid method...")
                sys.exit()
            if np.array_equal(y_pred, y[i]):
                count += 1
            if i > 0:
                t.set_description(f"{i} {count/i}")

def evaluate(model_file):

    # Load the model
    model = tf.keras.models.load_model(model_file)
    maxlen = model.layers[0].output_shape[1]

    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)
    if 'first' in model_file:
        names = import_names('first')
    if 'last' in model_file:
        names = import_names('last')
    x,y = build_training_set(names, vocab)
    padding_value = np.zeros((vocab.size,))

    model.evaluate(x,y)

