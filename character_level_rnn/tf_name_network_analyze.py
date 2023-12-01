#https://keras.io/examples/generative/lstm_character_level_text_generation/
from typing import List
from tensorflow import keras
import numpy as np
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import os
import sys
from datetime import datetime

def train(n_epochs, checkpoint_path, batch_size = None):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x, y, epochs=n_epochs, batch_size = batch_size, validation_split = 0.2, callbacks=[cp_callback])
    model.save(modelfile)
    return history

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


tr = 20              # Number of epochs to train (0 or False if you don't want to train)
cont = False           # Continue with previous weights (True or False)
batch_size = None
learning_rate=0.001

# Define the start and end characters of every name
START = "^"
STOP = "$"

# Load the shuffled names
shufflefile="data/shuffled_names.json"
vocabfile = 'tfweights/vocab_analyze.txt'
with open(shufflefile,'r') as f:
    names = json.load(f)
vocab = load_vocab(vocabfile)
print(vocab.w2i)

start_time = datetime.now()
print(f"Start time = ",start_time.strftime('%H:%M:%S'))

runs = ['firstnames','lastnames']
for run in runs:
    modelfile = f'tfweights/{run}.analyze.keras'
    checkpoint_path = f"tfweights/{run}.ckpt"

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
        if os.path.isfile(modelfile):
            model = keras.models.load_model(modelfile)
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
        batch_size = None
        history = train(tr, checkpoint_path, batch_size)

    print(history)
    # model.compile()
    # results = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print(results)

end_time = datetime.now()
difference = end_time - start_time
print(f"Total run time = ",difference)

