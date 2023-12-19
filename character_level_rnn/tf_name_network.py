# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

import tensorflow as tf
import numpy as np
import pandas as pd
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import os
import sys
from datetime import datetime
import tqdm
from scratch_name_network import import_names
import matplotlib.pyplot as plt


class NameModel(tf.keras.Model):
    def __init__(self, vocab, maxlen, hidden_dim = 32):
        super().__init__()
        self.padding_value = np.zeros((vocab.size,))
        self.maxlen = maxlen
        self.vocab_size = vocab.size
        self.hidden_dim = hidden_dim
        self.state = tf.zeros(shape = (1,self.hidden_dim))
        self.mask = tf.keras.layers.Masking(mask_value=self.padding_value, input_shape=(self.maxlen, self.vocab_size))
        self.RNN =  tf.keras.layers.SimpleRNN(self.hidden_dim, return_state = True)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, inputs, states = None, return_state=False, training=False):
        if training:
            x = inputs
            x = self.mask(x, training=True)
            x, _ = self.RNN(x, training=True)
            x = self.dense(x, training=True)
        else:
            x = inputs
            x = self.mask(x, training=False)
            x, self.state = self.RNN(x, training=False, initial_state = self.state)
            x = self.dense(x, training=False)
        return x

    def reset_state(self):
        self.state = tf.zeros(shape = (1,self.hidden_dim))

# class MemoryRNN(tf.keras.layers.Layer):
#     def __init__(self, hidden_dim=32, return_state = False):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.rnn_layer = tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=True, return_state=True)
#         self.initial_state = None
#         self.test = 500

#     def call(self, inputs, reset_state = False, states = None, return_state = False, training = False):


#         if training:
#             x, hidden_state = self.rnn_layer(inputs)
#             if return_state:
#                 # This line takes advantage of the fact that the hidden_state
#                 # of a simpleRNN layer is equal to the output with return_sequences = False
#                 # So we swap them when we output them.
#                 return x[:,-1,:], x
#             else:
#                 return x

#         else:
#             print(self.test)
#             sys.stdout.flush()
#             self.test += 1
#             # tf.print(self.states)

#             if reset_state or self.initial_state is None:
#                 self.initial_state = tf.zeros((tf.shape(inputs)[0], self.hidden_dim))

#             states = self.initial_state

#             outputs_ta = tf.TensorArray(tf.float32, size=tf.shape(inputs)[1])
#             for i in tf.range(tf.shape(inputs)[1]):
#                 x, states = self.rnn_layer(inputs[:, i:i+1, :], initial_state=states)
#                 outputs_ta = outputs_ta.write(i, x)

#             self.initial_state = states
            
#             # tf.print(self.initial_state)

#             if return_state:
#                 return states, states
#             else:
#                 return x

#     def reset_state(self):
#         self.initial_state = None

# class MemoryRNN(tf.keras.layers.Layer):
#     def __init__(self, hidden_dim=32):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.rnn_layer = tf.keras.layers.SimpleRNN(hidden_dim, return_sequences=True, return_state=True)

#     def call(self, inputs, states=None, training=False):
#         # if states is None:
#         #     states = tf.zeros((tf.shape(inputs)[0], self.hidden_dim))

#         all_states = []
#         outputs_ta = tf.TensorArray(tf.float32, size=tf.shape(inputs)[1])

#         for i in tf.range(tf.shape(inputs)[1]):
#             x, states = self.rnn_layer(inputs[:, i:i+1, :], initial_state=states)
#             # all_states.append(states)
#             outputs_ta = outputs_ta.write(i, x)

#         outputs = outputs_ta.stack()

#         # if return_state:
#         return outputs, all_states
#         # else:
#             # return outputs

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

class OutputHistory(tf.keras.callbacks.Callback):
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

def scheduler(epoch, lr):
    if epoch < 25:
        return 0.01
    elif epoch < 50:
        return 0.001
    else:
        return 0.0005

# Use the trained network to generate a new name
def generate(model, vocab, verbose = False):
    # Start with our starting character
    string = START
    x = np.zeros((1, maxlen, vocab.size))
    x[0, 0, vocab.w2i[START]] = 1.0

    if verbose: print(START,end="")
    # Encode the starting character
    for t in range(1,maxlen):
        # Generate the next character
        probabilities = model(x)[0].numpy()
        next_letter = vocab.i2w[sample_from(probabilities)]
        string += next_letter
        if verbose: print(next_letter,end="")
        sys.stdout.flush()
        # If this is our STOP character, we're done
        if string[-1] == STOP:
            if verbose: print("")
            model.reset_state()
            return string[1:-1]
        # If not, then add this to our string and
        # go back through the loop again.
        x[0, t, vocab.w2i[string[t]]] = 1.0
    # If we get here, it means we hit our max length
    # So return the string without the start character
    model.reset_state()
    return string[1:]

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

def train_network(tr = [0,0], 
                  gn = 20,
                  learning_rate = 0.01, 
                  batch_size = None, 
                  cont = False):
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    tf.random.set_seed(0)
    np.random.seed(0)
    # Define the start and end characters of every name
    global START, STOP, maxlen, fig
    START = "^"
    STOP = "$"

    vocabfile = f'finalweights/vocab.txt'  # Where to save the vocab file

    #Import the names from the file
    namesfile="data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    names = {}
    names['firstnames'] = [(START + player.firstname + STOP) for player in players if player.firstname is not None]
    names['lastnames'] = [(START + player.lastname + STOP) for player in players]
    suffixes = [player.suffix for player in players]

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
        # checkpoint_path = f"tfweights/{run}.keras"

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
                model = tf.keras.models.load_model(model_file)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            model = NameModel(vocab,maxlen)
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        if tr[r]:
            output_callback = OutputHistory(history_file)
            schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                                 monitor='accuracy',
                                                                 mode='max',
                                                                 save_best_only=True)
            history = model.fit(x, 
                                y, 
                                epochs=tr[r], 
                                batch_size = batch_size, 
                                callbacks = [output_callback,schedule_callback])
            model.save_weights(model_file)

        if gn:
            print(f"Generating {run}")
            for _ in range(gn):
                generated_names[run].append(generate(model,vocab,True))


    def random_suffix() -> str:
        suffix = np.random.choice(suffixes)
        return suffix if suffix is not None else ""

    for i in range(gn):
        print(generated_names['firstnames'][i],generated_names['lastnames'][i],random_suffix())

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total run time = ",difference)

def training_speed_test(n_epochs = 20, 
                        learning_rate = 0.01, 
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
                model = tf.keras.models.load_model(model_file)
            else:
                print("Model file does not exist")
                sys.exit()
        else:
            HIDDEN_DIM = 32
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Masking(mask_value=padding_value, input_shape=(maxlen, vocab.size)),
                    tf.keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                    tf.keras.layers.SimpleRNN(HIDDEN_DIM,),
                    tf.keras.layers.Dense(vocab.size, activation="softmax"),
                ]
            )
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss="categorical_crossentropy", 
                          optimizer=optimizer, 
                          metrics=['accuracy'])

        start_time = datetime.now()
        print(f"Start time = ",start_time.strftime('%H:%M:%S'))

        output_callback = OutputHistory(history_file, model_file)
        schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
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

def generation_test(n_players):
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    # Load in the names of the players and
    # set the filenames to be used to read/store
    # vocab info and network weights
    firstnames, lastnames, _ = import_names()
    firstnames = set([name[1:-1] for name in firstnames])
    lasstnames = set([name[1:-1] for name in lastnames])

    names = {'firstnames': firstnames,'lastnames':lastnames}

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    for key in names:
        # Set up neural network
        model_file = f'tfweights/{key}.keras'
        model = tf.keras.models.load_model(model_file)
        maxlen = model.layers[0].output_shape[1]

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

