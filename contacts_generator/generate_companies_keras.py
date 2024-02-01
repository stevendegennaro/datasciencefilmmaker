# Uses keras to create a character-level recurrent neural network
# Trains on the lists of compaines on the NYSE and NASDAQ
# then generates a new name one letter at a time

import os
import tensorflow as tf
from tensorflow import keras
from lite_model import LiteModel
import numpy as np
import pandas as pd
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import sys
from datetime import datetime
import pickle
from collections import Counter
import tqdm
import ast
import matplotlib.pyplot as plt
import operator

def import_names(shuffle = False):

    global maxlen, START, STOP
    START = "^"
    STOP = "#"

    exchanges = ['nyse','nasdaq']
    names = []
    for exchange in exchanges:
        filename = 'data/' + exchange +'.json'
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
        names.extend([(START + company['name'] + STOP) for company in data])
        names = list(set(names))

    vocab_file = 'tfweights/company_vocab.txt'
    if os.path.isfile(vocab_file):
        vocab = load_vocab(vocab_file)
    else:
        vocab = Vocabulary([c for name in names for c in name])
        save_vocab(vocab,'tfweights/company_vocab.txt')

    maxlen = max(len(string) for string in names)
    if shuffle: 
        np.random.shuffle(names)
    else:
        names.sort()

    return names, vocab

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

    global maxlen

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


# #################
# ### Callbacks ###
# #################

class OutputHistory(keras.callbacks.Callback):
    def __init__(self, history_file):
        self.history_file = history_file
        self.hist_df = pd.DataFrame({'epoch': pd.Series(dtype='int'),
                            'time': pd.Series(dtype='float'),
                            'loss': pd.Series(dtype='float'),
                            'accuracy': pd.Series(dtype='float')
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
    if epoch < 5:
        return 0.01
    elif epoch < 10:
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
    # tr = list of how many epochs to train
    # gn = number of names to generate when finished
    # batch_size = keras batch_size, i.e. the number of names to run through
    #   the network before updating. Each epoch still uses all names
    # cont = continue from the last run?

def run_network(tr: int = 1, 
                gn: int = 20,
                batch_size: int = 1000, 
                cont: bool = False,
                HIDDEN_DIM: int = 128,
                file_stem: str = '',
                model_type: str = 'RNN',
                scheduler: str = 'step_down_1000',
                learning_rate: float = 0.001) -> None:
 
    start_time = datetime.now()
    print(f"Start time = ",start_time.strftime('%H:%M:%S'))

    # Load the vocab file
    names, vocab = import_names()
    x,y,padding_value = build_training_set(names, vocab)

    generated_names = []

    model_file = f'tfweights/companies_{file_stem}.keras'
    history_file = f'tfweights/companies_{file_stem}.history'

    if cont:
        if os.path.isfile(model_file):
            model = keras.models.load_model(model_file)
            tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        else:
            print("Model file does not exist")
            sys.exit()
    else:
        model = create_model(padding_value,
                             maxlen,
                             vocab,
                             HIDDEN_DIM,
                             learning_rate = learning_rate,
                             model_type = model_type)

    print(model.summary())

    if tr:
        output_callback = OutputHistory(history_file)
        # Converts the name of a function into the function
        scheduler = globals()[scheduler]
        schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                              monitor='loss',
                                                              mode='min',
                                                              save_best_only=True)
        history = model.fit(x, 
                            y, 
                            epochs=tr, 
                            batch_size = batch_size,
                            callbacks = [output_callback,schedule_callback,checkpoint_callback])

        model.save(model_file)

    if gn:
        print(f"Generating...")
        for _ in range(gn):
            generated_names.append(generate(model,vocab))
        with open('tfweights/generated_names.txt','w') as f:
            json.dump(generated_names,f)


    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total run time = ",difference)

# ### Generate a list of n_players, first and last names
# ### Does not generate suffixes because at the moment
# ### I can't be bothered and they are not part of my analysis
# def generate_players(file_stem: str, n_players: int) -> dict[str: list[str]]:

#     global maxlen
#     names, vocab = import_names()

#     runs = ['firstnames','lastnames']
#     generated_names = {'firstnames':[],'lastnames':[]}

#     for run in runs:
#         print(f"Generating {run}...")
#         model_file = f'finalweights/keras/{run}_{file_stem}.keras'
#         model = keras.models.load_model(model_file)
#         maxlen = model.layers[0].output_shape[1]
#         model = LiteModel.from_keras_model(model)

#         x = np.zeros((1, maxlen, vocab.size))
#         x[0, 0, vocab.w2i[START]] = 1.0

#         with tqdm.trange(n_players) as t:
#             for _ in t:
#                 generated_names[run].append(generate(model,vocab))

#     return generated_names




### Tests how long it takes to generate n_names names
### Then calculates how many of those names are already
### in the first names list. Repeats for last names. Returns
### a dictionary with lists of the duplicates
def generation_test(file_stem: str, n_names: int) -> \
        tuple[list[str],list[str],float,float]:

    global maxlen
    # tf.compat.v1.disable_eager_execution()
    # Load in the names of the players and vocab
    names, vocab = import_names()
    names = set([name[1:-1] for name in names])

    # Set up neural network
    model_file = f'tfweights/companies_{file_stem}.keras'
    model = keras.models.load_model(model_file)
    maxlen = model.layers[0].output_shape[1]
    model = LiteModel.from_keras_model(model)

    start_time = datetime.now()
    print(f"Generating names",start_time.strftime('%H:%M:%S'))

    generated_names = []
    for _ in tqdm.tqdm(range(n_names)):
        generated_names.append(generate(model, vocab))

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total generation time = ",difference)

    count = 0
    duplicates = []
    print("Finding duplicates...")
    for name in tqdm.tqdm(generated_names):
        if name in names:
            count += 1
            duplicates.append(name)

    percent_recreated = count/n_names*100

    print(f"{count} names were already in the list ({percent_recreated}%)")

    return generated_names, duplicates, percent_recreated, difference

# def manual_accuracy_test(model_file: str, method:str = 'argmax') -> None:

#     names, vocab = import_names()
#     if 'first' in model_file:
#         x, y, padding_value = build_training_set(names['firstnames'], vocab)
#     elif 'last' in model_file:
#         x, y, padding_value = build_training_set(names['lastnames'], vocab)
        
#     # Set up the model
#     model = keras.models.load_model(model_file)
#     # lmodel = model
#     model = LiteModel.from_keras_model(model)

#     count = 0
#     with tqdm.trange(len(x)) as t:
#         for i in t:
#             probabilities = model.predict(x[[i]])[0]
#             y_pred = np.zeros(vocab.size, dtype=np.float32)
#             if method == 'argmax':
#                 y_pred[np.argmax(probabilities)] = 1.0
#             elif method == 'sample_from':
#                 y_pred[np.random.choice(len(probabilities),p=probabilities)] = 1.0
#             else:
#                 print("Invalid method...")
#                 sys.exit()
#             if np.array_equal(y_pred, y[i]):
#                 count += 1
#             if i > 0:
#                 t.set_description(f"{i} {count/i}")

# def evaluate(model_file: str) -> None:

#     # Load the model
#     model = keras.models.load_model(model_file)

#     names, vocab = import_names()
#     if 'first' in model_file:
#         x, y, padding_value = build_training_set(names['firstnames'], vocab)
#     elif 'last' in model_file:
#         x, y, padding_value = build_training_set(names['lastnames'], vocab)
    
#     model.evaluate(x,y)


### Takes a list of letters that may contain duplicates
### Returns the maximum accuracy one could acheive by
### choosing a letter at random according to its frequency in the list
### and then "predicting" that letter, also according to the
### frequency in the list. Put another way, if you draw two 
### letters randomly from the list (with replacement),
### what are the odds that you draw the same letter twice?
def predict_accuracy(letters_list):
    count = Counter(letters_list)
    probs = np.array(list(count.values()))
    sum_of_squares = np.sum(probs**2)
    square_of_sum = np.sum(probs)**2
    return sum_of_squares / square_of_sum

# What if instead of drawing from the list of letter with 
# frequency given by the list, we instead always draw
# the most frequent letter in the list (or if there is a
# tie, we choose at random from among the ties)
def get_most_likely(letters_list):
    count = dict(Counter(letters_list))
    most_likely = np.random.choice([key for key, value in count.items() if value == max(count.values())])
    prob = count[most_likely]/sum(count.values())
    return prob

### Calculate the theoretical maximum accuracy of the network, 
### given the list of names, choosing a letter at random from
### among the possible choices for that input, with frequency
### given by their frequency in the list
def calculate_max_accuracy(method = 'sample_from', dict_file:str = None) -> float:

    # Import the names and stick them in a list
    names, vocab = import_names()
    
    print("Building inputs and targets...")
    inputs = []
    targets = []
    for name in names:
        for i in range(1,len(name)):
            inputs.append(name[:i])
            targets.append(name[i])
    

    print("Calculating input freqencies...")
    input_frequency = Counter(inputs)
    for k in input_frequency.keys():
        input_frequency[k] /= len(inputs)

    if os.path.isfile(dict_file):
        with open(dict_file,'r') as f:
            set_dict = json.load(f)
    else:
        print("Building set_dict...")
        set_dict = {k: [] for k in input_frequency.keys()}
        for k in tqdm.tqdm(set_dict.keys()):
            for i,item in enumerate(inputs):
                if item == k:
                    set_dict[k].append(targets[i])

        with open(dict_file,'w') as f:
            json.dump(set_dict,f)

    print("Calculating accuracy...")
    accuracy = 0.0
    for k in tqdm.tqdm(set_dict.keys()):
        if method == 'argmax':
            accuracy += input_frequency[k] * get_most_likely(set_dict[k])
        elif method == 'sample_from':
            accuracy += input_frequency[k] * predict_accuracy(set_dict[k])
        else:
            print("Invalid method")
            sys.exit()
        
    print(accuracy)

    return set_dict # Just in case

# def average_length_before_unique(filename = 'tfweights/num_possibles.txt'):
#     names, vocab = import_names()
#     names_df = pd.DataFrame(names,columns = ['name'])
#     # names_df['Name'] = names_df['Name'].str[1:-1]
#     names_df.sort_values('name',inplace = True)
#     num_possibles = pd.DataFrame({'string': pd.Series(dtype='str'),
#                                         'n': pd.Series(dtype='int')})
#     num_possibles.to_csv(filename,index=False)
#     with tqdm.tqdm(names_df['name']) as t:
#         for name in t:
#             t.set_description(f"{name}")
#             for i in range(1,len(name)-1):
#                 string = name[:i]
#                 if not num_possibles['string'].isin([string]).any():
#                 #     n = len(names_df[names_df['name'].str.startswith(string)])
#                 #     num_possibles.loc[len(num_possibles.index)] = [string,n]

#                     new_n = pd.DataFrame({'string':[string],
#                                           'n':[len(names_df[names_df['name'].str.startswith(string)])]})
#                     new_n.to_csv(filename,mode='a',index=False,header = False)
#                     num_possibles.loc[len(num_possibles.index)] = new_n
    
#     return num_possibles



def average_length_before_unique(filename = 'tfweights/num_possibles.txt'):
    names, vocab = import_names()
    names_df = pd.DataFrame(names,columns = ['name'])
    names_df.sort_values('name',inplace = True)
    seen_so_far = []
    num_possibles = []

    if os.path.isfile(filename):
        print("Loading num_possibles from file...")
        with open(filename,"r") as f:
            for line in f: 
                line = ast.literal_eval(line.strip()) #or some other preprocessing
                num_possibles.append(line)

    else:
        with open(filename,"w") as f:
            with tqdm.tqdm(names_df['name']) as t:
                # For each name in the list
                for name in t:
                # for name in  names_df['name']:
                    t.set_description(f"{name[:10]}, {len(num_possibles)}")
                    # For each substring in the name up to len(name)
                    for i in range(1,len(name)):
                        substring = name[:i]
                        # If we haven't seen this substring thus far
                        if substring not in seen_so_far:
                            # Add it to the list of things we've seen
                            seen_so_far.append(substring)
                            # Find out how long it is
                            length = len(substring)
                            # Find out how many possibilities there are for the next letter
                            n = len(names_df[names_df['name'].str.startswith(substring)])
                            # Keep track of number of possibles vs length of string
                            f.write(f"{[substring,length,n]}\n")

    num_df = pd.DataFrame(num_possibles,columns = ['substring','length','n'])

    return num_df

def generated_names_final():
    with open('tfweights/generated_names.txt','r') as f:
        generated_names = [line.rstrip() for line in f]

    names,vocab = import_names()
    real_names = [name[1:-1] for name in names]

    final_list = [name for name in generated_names if name not in real_names]

    with open('tfweights/generated_names_final.txt','w') as f:
        for name in final_list:
            f.write(f'{name}\n')

    print(len(generated_names),len(final_list))

def plot_vocab_freqency():
    names, vocab = import_names()
    letters = dict(Counter([c for name in names for c in name]))
    n_letters = sum(letters.values())
    letters = {letter: value / n_letters for letter, value in letters.items()}
    letters = dict(sorted(letters.items(), key=operator.itemgetter(1),reverse=True))

    fig,ax = plt.subplots(figsize=(8,4),layout='constrained')
    ax.bar(letters.keys(), letters.values())
    ax.set_title('Frequencies of Letters in Company Names')
    ax.set_xlabel('Letter')
    ax.set_ylabel('Frequency')
    plt.show()


