import json
from typing import List, Tuple, Dict
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
from collections import Counter


# Import the names from the file,
# add the START and STOP characters,
# and put into separate lists for first and last names and suffixes
# Output to a tuple
def import_names(namesfile: str = "data/all_names.json") -> Tuple:
    # Define the start and end characters of every name
    global START, STOP
    START = "^"
    STOP = "$"

    with open(namesfile,"r") as f:
        entries = json.load(f)
    players = [Player(entry) for entry in entries]
    firstnames = [(START + player.firstname + STOP) \
                    for player in players if player.firstname is not None]
    lastnames = [(START + player.lastname + STOP) for player in players]
    suffixes = [player.suffix for player in players]

    return firstnames, lastnames, suffixes

def calculate_accuracy(model: Model, 
                       test_names: List, 
                       vocab: Vocabulary,
                       method: str = 'sample_from') -> float:
    n_correct = 0
    count = 0
    with tqdm.tqdm(test_names) as t:
        for name in t:
            model.layers[0].reset_hidden_state()
            model.layers[1].reset_hidden_state()
            for letter,next_letter in zip(name,name[1:]):
                inputs = vocab.one_hot_encode(letter)
                targets = vocab.one_hot_encode(next_letter)
                predicted = model.forward(inputs)
                probabilities = softmax(predicted)
                if method == 'sample_from':
                    next_char_predicted = vocab.i2w[sample_from(probabilities)]
                elif method == 'argmax':
                    next_char_predicted = vocab.i2w[np.argmax(probabilities)]
                else:
                    print(f"Method '{method}' not valid. Exiting...")
                    sys.exit()
                if next_char_predicted == next_letter: 
                    n_correct += 1
                count += 1
            t.set_description(f"{n_correct/count}")
    accuracy = n_correct / count
    return accuracy

def plot_history(history: pd.DataFrame, color:str = "black") -> None:
    ax = fig.gca()
    ax.cla()
    ax.plot(history['Time'],history["Accuracy"], color=color)
    plt.draw()
    plt.pause(0.1)

#### Main training loop, called by other functions
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
         ) -> None:

    start_time = datetime.now()
    print(f"Training start time = ",start_time.strftime('%H:%M:%S'))

    max_accuracy = 0.0
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
        # Otherwise, calculate the accuracy using a subset of the trianing data
        if test_names:
            accuracy = calculate_accuracy(model, test_names, vocab)
        else:
            accuracy = calculate_accuracy(model, train_names, vocab)

        # Output various parameters to the screen
        batch_time = datetime.now() - start_time
        print(f"Epoch: {epoch}  Epoch Loss: {epoch_loss}  Accuracy: {accuracy*100:.2f}  Elapsed Time: {batch_time}")
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
        # Save the model only if the accuracy is higher
        if accuracy >= max_accuracy:
            max_accuracy = accuracy
            save_weights(model,weight_file)

    end_time = datetime.now()
    difference = end_time - start_time
    print(f"Total training time = ",difference)

def generate(model: Model, 
             vocab: Vocabulary, 
             max_len: int = 160) -> str:

    # Reset the state of the hidden layers
    model.layers[0].reset_hidden_state()
    model.layers[1].reset_hidden_state()
    # Start with a list containing only the START character
    output = [START]

    # Loop to generate the name, one letter at a time
    # until we hit our STOP character or maximum length
    while output[-1] != STOP and len(output) < max_len:
        # One-hot encode the previous letter 
        this_input = vocab.one_hot_encode(output[-1])
        # Predict the next letter
        predicted = model.forward(this_input)
        probabilities = softmax(predicted)
        next_char_id = sample_from(probabilities)
        # Add the letter to the end of the output and go back
        output.append(vocab.get_word(next_char_id))

    # Return the output minus START and STOP characters
    return ''.join(output[1:-1])

def get_vocab(names: List) -> Vocabulary:
    # create a Vocabulary object from a list of names
    vocab = Vocabulary([c for name in names for c in name])
    return vocab

def create_model(vocab: Vocabulary,
                 HIDDEN_DIM: int = 32, 
                 learning_rate: float = 0.01, 
                 momentum: float = 0.9) -> Model:
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
    # tr =  train the network
    # cont = continue training from previous weights
    # gen = generate names when done training
    # batch_size = subset of entire data to train on per epoch
    # plot = plot the loss and accuracy after each epoch at runtime
def run_network(which_names: str = 'lastnames', 
                  tr: bool = True, 
                  learning_rate = 0.01,
                  cont: bool = False, 
                  gen: bool = True,
                  seed: bool = None,
                  batch_size: int = None,
                  n_epochs: int = 10,
                  plot: bool = True) -> None:

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

    model.optimizer.lr = learning_rate

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

        with open('weights/generated_test.txt',"a") as f:
            json.dump(generated_names,f)

#### Generate a batch of first and last names 
#### based on final weights from the training sessions
def generate_players(n_players: int) -> List:
    
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

    generated_names = []
    for i in range(len(generated_first_names)):
        generated_names.append(generated_first_names[i],generated_last_names[i],random_suffix())

    return generated_names

def training_speed_test(n_epochs: int, 
                        learning_rate: float = 0.01, 
                        batch_size: int = None, 
                        cont: bool = False,
                        momentum: float = 0.9) -> None:

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


### Tests how long it takes to generate n_players names
### Then calculates how many of those names are already
### in the names list. Repeats for last names. Returns a
### dictionary with lists of the duplicates
def generation_test(n_players: int) -> dict:
    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)
    
    # Load in the names of the players
    firstnames, lastnames, _ = import_names()
    firstnames = set([name[1:-1] for name in firstnames])
    lasstnames = set([name[1:-1] for name in lastnames])
    names = {'firstnames': firstnames,'lastnames':lastnames}
    duplicates = {}

    for key in names:
        weight_file = f"finalweights/{key}_weights.txt"

        # Set up neural network
        model = create_model(vocab)
        load_weights(model,weight_file)

        start_time = datetime.now()
        print(f"Generating {key} names",start_time.strftime('%H:%M:%S'))
        generated_names = []

        for _ in tqdm.tqdm(range(n_players)):
            generated_names.append(generate(model, vocab))

        end_time = datetime.now()
        difference = end_time - start_time
        print(f"Total generation time = ",difference)

        ### This part calculates how many of the names
        ### that we just generated are already in our
        ### list of names.
        duplicates[key] = []
        count = 0
        for name in tqdm.tqdm(generated_names):
            if name in names[key]:
                count += 1
                duplicates[key].append(name)

        print(f"{count} names were already in the list ({count/n_players*100}%)")

    return duplicates


### Tests how long it takes to generate n_players names
### as a function of the length of the name
def generation_timing_test(n_players: int) -> pd.DataFrame:
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

#### Tests the frequency of generated first names vs the frequency
#### in the original data set, plotted by length of the name
    # duplicates = dictionary of generated names 
    # with keys 'firstnames' and 'lastnames', which are
    # generated by generation_test()
def generated_frequency_test(duplicates: Dict):

    df = pd.DataFrame.from_dict(Counter(duplicates['firstnames']),orient = 'index').reset_index()
    df.columns = ['Name','Generated Count']
    df.sort_values(by = 'Generated Count', ascending = False,inplace = True)
    firstnames, _, _ = import_names()
    firstnames = [name[1:-1] for name in firstnames]
    firstnames_df = pd.DataFrame.from_dict(Counter(firstnames),orient = 'index').reset_index()
    firstnames_df.columns = ['Name','Count']
    firstnames_df.sort_values(by = 'Count', ascending = False,inplace = True)
    merged_df = pd.merge(firstnames_df,df)
    merged_df['Length'] = merged_df['Name'].str.len()
    fig = plt.figure(figsize=(5,7))
    nrows = 4
    ncols = 2
    axes = [fig.add_subplot(nrows, ncols, r * ncols + c + 1) for r in range(0, nrows) for c in range(0, ncols)]

    for i,ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(merged_df[merged_df['Length'] == i + 2]['Count'],
                   merged_df[merged_df['Length'] == i + 2]['Generated Count'],
                   marker = '.',
                   c = 'black', label = f"{i+2}")
        leg = ax.legend(loc = 'lower right',handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0,left=0.1, right=0.9, top=0.9, bottom=0.05)
    fig.supxlabel("Frequency of Name in Original List")
    fig.supylabel("Frequency of Name in Generated List")
    fig.suptitle("Frequency Comparison of First Names\nin Generated List vs Original")
    plt.show()

### Find the accuracy of a trained network using either
### 'sample_from' or 'argmax'
def manual_accuracy_test(which:str = 'first', method:str = 'argmax') -> None:

    global START, STOP, maxlen
    START = "^"
    STOP = "$"

    print(f"Calculating accuracy for {which}names...")

    # Load in the names of the players and
    # set the filenames to be used to read
    # vocab info and network weights
    firstnames, lastnames, suffixes = import_names()
    if which == 'last':
        names = lastnames
    elif which == 'first':
        names = firstnames
    else:
        assert False, 'Must choose firstnames or lastnames'
    weight_file = f"finalweights/{which}names_weights.txt"

    vocab_file = f"finalweights/vocab.txt"
    vocab = load_vocab(vocab_file)

    # Create model
    print(f"Loading weights from {weight_file}...")
    model = create_model(vocab)
    load_weights(model,weight_file)

    accuracy = calculate_accuracy(model, names, vocab, method)

    print(f"Total accuracy is {accuracy*100:.2f}% for {which}names using method '{method}'")

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
def calculate_max_accuracy() -> float:

    # Import the names and stick them in a dictionary
    firstnames, lastnames, suffixes = import_names()
    names = {'firstnames': firstnames, 'lastnames': lastnames}
    
    for run in names.keys():
        print(run.title())
        print("Building inputs and targets...")
        inputs = []
        targets = []
        for name in names[run]:
            for i in range(1,len(name)):
                inputs.append(name[:i])
                targets.append(name[i])
        
        print("Calculating input freqencies...")
        input_frequency = Counter(inputs)
        for k in input_frequency.keys():
            input_frequency[k] /= len(inputs)

        print("Building set_dict...")
        set_dict = {k: [] for k in input_frequency.keys()}
        for k in tqdm.tqdm(set_dict.keys()):
            for i,item in enumerate(inputs):
                if item == k:
                    set_dict[k].append(targets[i])

        print("Calculating accuracy...")
        accuracy = 0.0
        for k in tqdm.tqdm(set_dict.keys()):
            accuracy += input_frequency[k] * predict_accuracy(set_dict[k])
            
        print(accuracy)

### Calculate the theoretical maximum accuracy of the network, 
### given the list of names, always choosing the most likely
### letter (i.e. the mode) for a given input
def calculate_max_likelihood() -> float:

    # Import the names and stick them in a dictionary
    firstnames, lastnames, suffixes = import_names()
    names = {'firstnames': firstnames, 'lastnames': lastnames}
    
    for run in names.keys():
        print(run.title())
        print("Building inputs and targets...")
        inputs = []
        targets = []
        for name in names[run]:
            for i in range(1,len(name)):
                inputs.append(name[:i])
                targets.append(name[i])
        
        print("Calculating input freqencies...")
        input_frequency = Counter(inputs)
        for k in input_frequency.keys():
            input_frequency[k] /= len(inputs)

        print("Building set_dict...")
        set_dict = {k: [] for k in input_frequency.keys()}
        for k in tqdm.tqdm(set_dict.keys()):
            for i,item in enumerate(inputs):
                if item == k:
                    set_dict[k].append(targets[i])

        print("Calculating accuracy...")
        accuracy = 0.0
        for k in tqdm.tqdm(set_dict.keys()):
            accuracy += input_frequency[k] * get_most_likely(set_dict[k])
            
        print(accuracy)




