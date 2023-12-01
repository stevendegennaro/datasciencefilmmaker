# Uses keras to create a character-level recurrent neural network
# Trains separately on the first and last names of every 
# Major League Baseball player in history, then generates a new name
# by generating a new first name and a new last name one letter
# at a time, then adding a random suffix from the list of suffixes

from tensorflow import keras
import numpy as np
from player import Player
import json
from dsfs_vocab import Vocabulary, save_vocab, load_vocab
import os
import sys
from datetime import datetime

# Takes a list of weights and returns a random
# index chosen based on the weights
def sample_from(weights):
    total = sum(weights)
    rnd = total * np.random.random()
    for i,w in enumerate(weights):
        rnd -= w 
        if rnd <= 0: return i

# Train the nerual network
def train(n_epochs, batch_size = None):
    # We train each epoch separately so that we can generate a new name 
    # after each epoch to see how well the network is doing
    for epoch in range(n_epochs):
        model.fit(x, y, epochs=1, batch_size = batch_size)
        print()
        print("Generating text after epoch: %d" % epoch)
        print(generate())
        print("Saving model")
        # Save the model after every epoch in case we need to
        # interrupt and restart
        model.save(modelfile)

# Use the trained network to generate a new name
def generate():
    # Start with our starting character
    string = START
    # Keep going until we hit our ending character 
    # or we reach 100 characters total
    while string[-1] != STOP and len(string) < 100:
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
    modelfile = f'tfweights/{run}.keras'

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

