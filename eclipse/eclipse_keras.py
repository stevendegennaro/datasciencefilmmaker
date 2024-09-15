import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import keras
from import_eclipse_data import import_solar_eclipse_data
from collections import Counter
import os
import pickle


def interval_runs_histogram(interval_runs):
    count = Counter(interval_runs)
    count_df = pd.DataFrame.from_dict(count, orient='index').astype(np.int32)
    count_df.index = pd.MultiIndex.from_tuples(count_df.index)
    count_df.sort_index(inplace=True)
    count_df.columns = ['Count']
    count_df.unstack().plot(kind='bar')
    plt.show()


# def tokenize_time_to_next(time_to_next):
#     time_to_next = time_to_next.copy()
#     time_to_next[time_to_next > 170] = 200
#     time_to_next[(time_to_next < 170) & (time_to_next > 140)] = 100
#     time_to_next[time_to_next < 99] = 0
#     time_to_next /= 100
#     time_to_next = time_to_next.astype(np.int32)
#     return time_to_next


def count_interval_runs(data):
    time_to_next = np.array(data['Time to Next'])[3:-3]
    time_to_next = tokenize_time_to_next(time_to_next)

    # fig,ax = plt.subplots(1,figsize=[7,1])
    # ax.set_xlim([10,100])
    # plt.plot(time_to_next,'.')
    # plt.show()
    # print(time_to_next)
    interval_runs = []
    i = 0
    while i < len(time_to_next) - 1:
        value = time_to_next[i]
        count = 0
        j = i
        while j < len(time_to_next) and time_to_next[j] == time_to_next[i]:
            j += 1
            count += 1
        i = j
        interval_runs.append((value,count))

    # plt.plot(interval_runs[:,1],'.k')
    # plt.show()

    interval_runs_histogram(interval_runs)

def one_hot_encode_data(time_to_next, length_of_prediction_vector):
    '''
        Build the training set. Takes the list of intervals and
        creates matrices of one-hot encoded vectors where
        the length of the inputs is length_of_prediction_vector
        and the target is the following intervals
    '''

    time_to_next = np.array(time_to_next)
    n_training_samples = len(time_to_next) - length_of_prediction_vector

    print(f"Building training set...")
    inputs = []
    targets = []
    for i in range(n_training_samples):
        inputs.append(time_to_next[i:i + length_of_prediction_vector])
        targets.append(time_to_next[i + length_of_prediction_vector])

    # One-hot encode inputs and targets and put into np array
    print("One-hot encoding inputs and targets")
    x = np.zeros((n_training_samples, length_of_prediction_vector, 3), dtype=np.float32)
    y = np.zeros((n_training_samples, 3), dtype=np.float32)

    for i in range(n_training_samples):
        for t in range(length_of_prediction_vector):
            x[i, t, inputs[i][t]] = 1
        y[i, targets[i]] = 1

    return x,y

def create_model(length_of_prediction_vector: int,
                 HIDDEN_DIM: int,
                 learning_rate: float,
                 model_type: str = 'RNN') -> keras.models:
    if model_type == 'RNN':
        model = keras.Sequential(
            [
                keras.Input(shape=(length_of_prediction_vector, 3)),
                # keras.layers.SimpleRNN(HIDDEN_DIM,return_sequences=True),
                keras.layers.SimpleRNN(HIDDEN_DIM,),
                keras.layers.Dense(3, activation="softmax"),
            ]
        )
    elif model_type == 'LSTM':
        model = keras.Sequential(
            [
                keras.Input(shape=(length_of_prediction_vector, 3)),
                # keras.layers.LSTM(HIDDEN_DIM,return_sequences=True),
                keras.layers.LSTM(HIDDEN_DIM,),
                keras.layers.Dense(3, activation="softmax"),
            ]
        )
    else:
        print(f"Invalid model type '{model_type}'. Exiting...")
        sys.exit()

    optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def eclipse_network(length_of_prediction_vector = 1000,
                    n_epochs = 10,
                    batch_size = 100,
                    N_HIDDEN = 32,
                    lr = 0.001,
                    model_type = 'RNN'):

    # Import raw data
    data = import_solar_eclipse_data()
    train_data = data[(data['Year']<=2023) | ((data['Year'] == 2024) & (data['Month'] == 4))]
    # time_to_next = np.array(train_data['Time to Next'])

    # print(data)
    # return
    # Convert to one-hot encoded numpy arrays
    x, y = one_hot_encode_data(train_data['Level'], length_of_prediction_vector)

    model_file = f'models/eclipse.{length_of_prediction_vector}.{model_type}.{N_HIDDEN}.keras'
    if os.path.isfile(model_file):
        model = keras.models.load_model(model_file)
        keras.backend.set_value(model.optimizer.learning_rate, lr)
        print("Model loaded")
    else:
        model = create_model(length_of_prediction_vector,N_HIDDEN,lr,model_type)
        print("Model created")

    history = model.fit(x, 
                        y, 
                        epochs = n_epochs, 
                        batch_size = batch_size
                    )

    model.save(model_file)


# def evaluate_model(data, model_file, length_of_prediction_vector):
#     mo 
################
### Generate ###
################
# Use the trained network to generate a single new name
def generate_next(model, x):
    pass

def generation_plots(x,y,y_fit,log=False):

    y_true = np.argmax(y,axis=1)
    y_guess = np.argmax(y_fit,axis=1)
    correct = (y_guess == y_true)

    ## 
    fig1, axes1 = plt.subplots(3)
    fig2, ax2 = plt.subplots(1)
    fig3, ax3 = plt.subplots(1)
    labels = ['Low (29 days)','Med (58 days)','Hi (179 days)']
    percent_correct = np.zeros(3)
    guesses = np.zeros([3,3])
    for i in range(3):
        level = (y[:,i] == 1.0)
        axes1[2-i].hist(y_fit[level,i],bins=20)
        axes1[2-i].set_ylabel(labels[i])
        if log: axes1[2-i].set_yscale("log")
        percent_correct[i] = correct[y_true==i].sum()/(y_true==i).sum()
        for j in range(3):
            guesses[i,j] = (y_guess[y_true == i]==j).sum()/len(y_guess[y_true == i])
    axes1[2].set_xlabel('Probability')

    percent_correct = pd.DataFrame(percent_correct,columns=['% Correct'],index=["Low","Med","Hi"])
    percent_correct['% Incorrect'] = 1 - percent_correct['% Correct']
    percent_correct *= 100
    cmap = colors.LinearSegmentedColormap.from_list("", ["navy","royalblue","lightsteelblue"])
    percent_correct.plot(kind='bar',
                         stacked=True,
                         ax=ax2,
                         colormap=cmap,
                         legend=False)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.set_xlabel("True Value")
    ax2.set_ylabel("% Correct")
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.tick_params(axis='x', labelrotation=0)

    guesses = pd.DataFrame(guesses,
                           columns = ["Low","Med","Hi"],
                           index=["Low","Med","Hi"])
    guesses *= 100
    guesses.plot(kind='bar',
                 stacked=True,
                 ax=ax3,
                 colormap=cmap,
                 legend=False)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Guess")
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax3.tick_params(axis='x', labelrotation=0)
    ax3.set_xlabel("True Value")
    ax3.set_ylabel("Percent")
    plt.show()

    print(f"Overall Percent Correct = {correct.sum()/len(y)}%")

def generation_test(model_file, length_of_prediction_vector = 1000, log = True):

    # Import the model
    model = keras.models.load_model(model_file)

    pickle_file = f'{model_file[:-5]}pickle'
    if os.path.isfile(pickle_file):
        with open(pickle_file,'rb') as f:
            (x, y, y_fit) = pickle.load(f)

    else:
        # Import raw data
        data = import_solar_eclipse_data()
        time_to_next = np.array(data.loc[9561 - length_of_prediction_vector:,'Time to Next'])
        x,y = one_hot_encode_data(time_to_next,length_of_prediction_vector)

        # Predict the next eclipse
        y_fit = model.predict(x)

        with open(pickle_file,'wb') as f:
            pickle.dump((x, y, y_fit),f)

    generation_plots(x,y,y_fit,log)

    return model


def pure_generation_test(model_file, length_of_prediction_vector = 1000):

    # pickle_file = f'{model_file[:-5]}pickle'
    # if os.path.isfile(pickle_file):
    #     with open(pickle_file,'rb') as f:
    #         (x, y, y_fit) = pickle.load(f)

    # else:
    # Import raw data
    data = import_solar_eclipse_data()
    time_to_next = np.array(data.loc[9561 - length_of_prediction_vector:,'Time to Next'])
    x,y = one_hot_encode_data(time_to_next,length_of_prediction_vector)

    # Import the model
    model = keras.models.load_model(model_file)

    # Predict the next eclipse
    y_fit = model.predict(x)
 
#         with open(pickle_file,'wb') as f:
#             pickle.dump((x, y, y_fit),f)





# Predicting just the next eclipse:
# len(correct[correct == True])/len(correct)
# Out[124]: 0.9567821994009413

# In [147]: y_correct[1000:].sum()/len(y_correct[1000:])
# Out[147]: 0.955871353777113

# In [148]: y_correct[:1000].sum()/len(y_correct[:1000])
# Out[148]: 0.958

# In [152]: y_correct[y_true==2].sum()/len(y_correct[y_true==2])
# Out[152]: 0.9728296885354539

# In [153]: y_correct[y_true==1].sum()/len(y_correct[y_true==1])
# Out[153]: 0.9653916211293261

# In [154]: y_correct[y_true==0].sum()/len(y_correct[y_true==0])
# Out[154]: 0.8530465949820788





