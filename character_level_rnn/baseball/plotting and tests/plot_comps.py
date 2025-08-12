import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import json
from player import Player

def import_names(namesfile: str = "../data/all_names.json"):
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


def plot_single_history(history_file: str, 
                 y_column: str = 'accuracy', 
                 color:str = "black",
                 fig = None) -> None:

    hist_df = pd.read_csv(history_file)
    if fig:
        ax = fig.gca()
        ax.plot(hist_df['time'],hist_df[y_column], color=color)
        return fig
    else:
        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(hist_df['time'],hist_df[y_column], color=color)
        plt.show()
        return fig


def learning_rate_plot():
    history1 = pd.read_csv("weights/firstnames_0.01_500_history.txt")
    history2 = pd.read_csv("weights/firstnames_0.05_500_history.txt")
    history3 = pd.read_csv("weights/firstnames_0.005_500_history.txt")

    fig,ax = plt.subplots()
    ax.plot(history1['Time'],history1["Accuracy"],color="red", label="Learning Rate = 0.01")
    ax.plot(history2['Time'],history2["Accuracy"],color="black", label="Learning Rate = 0.05")
    ax.plot(history3['Time'],history3["Accuracy"],color="blue",label="Learning Rate = 0.005")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def batch_size_plot():
    history1 = pd.read_csv("weights/firstnames_0.01_500_history.txt")
    history2 = pd.read_csv("weights/firstnames_0.01_1000_history.txt")
    history3 = pd.read_csv("weights/firstnames_0.01_18187_history.txt")

    fig,ax = plt.subplots()
    ax.plot(history1['Time'],history1["Accuracy"],color="red", label="Batch Size = 500")
    ax.plot(history2['Time'],history2["Accuracy"],color="black", label="Batch Size = 1,000")
    ax.plot(history3['Time'],history3["Accuracy"],color="blue",label="Batch Size = 18,187")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def tf_batch_size_plot():
    history1 = pd.read_csv("tfweights/firstnames_0.001_1000.history")
    history2 = pd.read_csv("tfweights/firstnames_0.001_32.history")
    history3 = pd.read_csv("tfweights/firstnames_0.001_1001.history")

    fig,ax = plt.subplots()
    ax.plot(history2['time'],history2["val_accuracy"],color="black", label="Batch Size = None (32)")
    ax.plot(history1['time'],history1["val_accuracy"],color="red", label="Batch Size = 1000")
    ax.plot(history3['time'],history3["val_accuracy"],color="blue", label="Batch Size = 10000")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def tf_learning_rate_plot():
    history1 = pd.read_csv("tfweights/lastnames_0.001_1000.history")
    history2 = pd.read_csv("tfweights/lastnames_0.01_1000.history")
    # history3 = pd.read_csv("tfweights/firstnames_0.001_1001.history")

    fig,ax = plt.subplots()
    ax.plot(history2['time'],history2["val_accuracy"],color="black", label="Learning Rate = 0.001")
    ax.plot(history1['time'],history1["val_accuracy"],color="red", label="Learning Rate = 0.01")
    # ax.plot(history3['time'],history3["val_accuracy"],color="blue", label="Batch Size = 10000")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def momentum_plot():
    history1 = pd.read_csv("weights/firstnames_0.01_1000_history.txt")
    history2 = pd.read_csv("weights/firstnames_0.01_1000_0.5_history.txt")
    history3 = pd.read_csv("weights/firstnames_0.01_1000_0.1_history.txt")
 
    fig,ax = plt.subplots()
    ax.plot(history1['Time'],history1["Accuracy"],color="red", label="Momentum = 0.9")
    ax.plot(history2['Time'],history2["Accuracy"],color="black", label="Momentum = 0.5")
    ax.plot(history3['Time'],history3["Accuracy"],color="blue",label="Momentum = 0.1")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def firstname_frequency():

    firstnames, lastnames, suffixes = import_names()
    firstnames = [name[1:-1] for name in firstnames]
    lastnames = [name[1:-1] for name in lastnames]
    fn_count = Counter(firstnames)
    fn_df = pd.DataFrame.from_dict(fn_count,orient='index')
    fn_df.columns= ['count']
    fn_df.sort_values(by=['count'],ascending = False,inplace =  True)
    fn_df = fn_df.iloc[0:100]
    fn_df.plot(kind='bar', width=1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylabel('Count')
    ax.set_xlabel('Name')
    ax.get_legend().remove()
    ax.set_title("Frequency of Select First Names Among Baseball Players")
    plt.xticks(rotation=60)
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def longtail_frequency():

    firstnames, lastnames, suffixes = import_names()
    lastnames = [name[1:-1] for name in lastnames]
    firstnames = [name[1:-1] for name in firstnames]
    ln_count = Counter(lastnames)
    fn_count = Counter(firstnames)
    ln_df = pd.DataFrame.from_dict(ln_count,orient='index')
    fn_df = pd.DataFrame.from_dict(fn_count,orient='index')
    ln_df.columns= ['Last Names']
    fn_df.columns= ['First Names']
    ln_df.sort_values(by=['Last Names'],ascending = False,inplace =  True)
    fn_df.sort_values(by=['First Names'],ascending = False,inplace =  True)
    print(len(fn_df),len(ln_df))
    n_names = 200
    ln_df = ln_df.iloc[0:n_names]
    fn_df = fn_df.iloc[0:n_names]
    fig,ax = plt.subplots()
    fn_df.plot(ax = ax, kind='bar', width=1)
    ln_df.plot(ax = ax, kind='bar', width=1,color="red")
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.set_ylabel('Count')
    ax.set_title("Frequency of First and Last Names\nAmong MLB Baseball Players")
    ax.set_xlabel("Name")
    plt.show()

    return fn_df,ln_df

def firstnames_comparison_plot():
    history1 = pd.read_csv("weights/firstnames_0.01_500_history.txt")
    history2 = pd.read_csv("tfweights/firstnames_0.01_1000.history")

    fig,ax = plt.subplots()
    ax.plot(history1['Time'],history1["Accuracy"],color="black", label="Scratch Network")
    ax.plot(history2['time'],history2["val_accuracy"],color="red", label="Keras Network")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Networks Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def lastnames_comparison_plot():
    history1 = pd.read_csv("weights/lastnames_0.01_500_0.9_history.txt")
    # history1 = pd.read_csv("weights/lastnames_history.txt")
    history2 = pd.read_csv("tfweights/lastnames_0.01_1000.history")

    fig,ax = plt.subplots()
    ax.plot(history1['Time'],history1["Accuracy"],color="black", label="Scratch Network")
    ax.plot(history2['time'],history2["val_accuracy"],color="red", label="Keras Network")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Networks Trained on Last Names\n"
    ax.set_title(title)
    plt.show()

def neurons_comparison_plot_firstnames():
    history1 = pd.read_csv("tfweights/firstnames_32.history")
    history2 = pd.read_csv("tfweights/firstnames_128.history")

    fig,ax = plt.subplots()
    ax.plot(history1['time'],history1["val_accuracy"],color="black", label="32 Hidden Units")
    ax.plot(history2['time'],history2["val_accuracy"],color="red", label="128 Hidden Units")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Networks Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def neurons_comparison_plot_lastnames():
    history1 = pd.read_csv("tfweights/lastnames_32.history")
    history2 = pd.read_csv("tfweights/lastnames_128.history")

    fig,ax = plt.subplots()
    ax.plot(history1['time'],history1["val_accuracy"],color="black", label="32 Hidden Units")
    ax.plot(history2['time'],history2["val_accuracy"],color="red", label="128 Hidden Units")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Networks Trained on Last Names\n"
    ax.set_title(title)
    plt.show()

def lstm_comparison_plot_firstnames():
    history1 = pd.read_csv("finalweights/firstnames_RNN_32.history")
    history2 = pd.read_csv("finalweights/firstnames_RNN_128.history")
    history3 = pd.read_csv("tfweights/firstnames_LSTM.history")

    fig,ax = plt.subplots()
    ax.plot(history1['time'],history1["accuracy"],color="black", label="SimpleRNN (32)")
    ax.plot(history2['time'],history2["accuracy"],color="red", label="SimpleRNN (128)")
    ax.plot(history3['time'],history3["accuracy"],color="blue", label="LSTM")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Networks Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def lstm_comparison_plot_lastnames():
    history1 = pd.read_csv("finalweights/lastnames_32.history")
    history1_p2 = pd.read_csv("finalweights/lastnames_32_full.history")
    history2 = pd.read_csv("tfweights/lastnames_LSTM.history")

    fig,ax = plt.subplots()
    ax.plot(history1['time'],history1["val_accuracy"],color="black", label="SimpleRNN")
    ax.plot(history2['time'],history2["val_accuracy"],color="red", label="LSTM")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Keras Networks Trained on Last Names\n"
    ax.set_title(title)
    plt.show()

# lstm_comparison_plot_lastnames()
# lstm_comparison_plot_firstnames()

def accuracy_bar_chart(which = 0):
    keras_argmax = [59.0, 43.5]
    keras_sample_from = [48.0, 30.3]

    scratch_argmax = [51.5,37.4]
    scratch_sample_from = [39.6,24.6]

    theoretical_argmax = [62.7, 60.1]
    theoretical_sample_from = [54.1, 53.6]

    method = ("sample_from", "argmax")
    accuracy = {
        'Scratch Network': (scratch_sample_from[which], scratch_argmax[which]),
        'Keras Network': (keras_sample_from[which], keras_argmax[which]),
        'Theoretical Max': (theoretical_sample_from[which], theoretical_argmax[which]),
    }

    x = np.arange(len(method))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Comparison for {"Last" if which else "First"} Names')
    ax.set_xticks(x + width, method)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 100)

    plt.show()

def accuracy_bar_chart_new(which = 0):
    keras_argmax = [59.0, 43.7]
    keras_sample_from = [48.0, 30.3]

    scratch_argmax = [56.9,40.0]
    scratch_sample_from = [44.8,26.8]

    theoretical_argmax = [62.7, 60.1]
    theoretical_sample_from = [54.1, 53.6]

    method = ("sample_from", "argmax")
    accuracy = {
        'Scratch Network': (scratch_sample_from[which], scratch_argmax[which]),
        'Keras Network': (keras_sample_from[which], keras_argmax[which]),
        'Theoretical Max': (theoretical_sample_from[which], theoretical_argmax[which]),
    }

    x = np.arange(len(method))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Comparison for {"Last" if which else "First"} Names')
    ax.set_xticks(x + width, method)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 100)

    plt.show()


def accuracy_bar_chart_128(which = 0):

    keras_32_argmax = [59.0, 43.7]
    keras_32_sample_from = [48.0, 30.3]

    keras_128_argmax = [62.1, 54.6]
    keras_128_sample_from = [52.5, 43.9]

    scratch_argmax = [56.9,40.0]
    scratch_sample_from = [44.8,26.8]

    theoretical_argmax = [62.7, 60.1]
    theoretical_sample_from = [54.1, 53.6]

    method = ("sample_from", "argmax")
    accuracy = {
        'Scratch Network': (scratch_sample_from[which], scratch_argmax[which]),
        'Keras Network (HIDDEN_DIM = 32)': (keras_32_sample_from[which], keras_32_argmax[which]),
        'Keras Network (HIDDEN_DIM = 128)': (keras_128_sample_from[which], keras_128_argmax[which]),
        'Theoretical Max': (theoretical_sample_from[which], theoretical_argmax[which]),
    }

    x = np.arange(len(method))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Comparison for {"Last" if which else "First"} Names')
    ax.set_xticks(x + 2*width, method)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 100)

    plt.show()


def accuracy_bar_chart_LSTM(which = 0):

    keras_32_argmax = [59.0, 43.7]
    keras_32_sample_from = [48.0, 30.3]

    keras_128_argmax = [62.1, 54.6]
    keras_128_sample_from = [52.5, 43.9]

    scratch_argmax = [56.9,40.0]
    scratch_sample_from = [44.8,26.8]

    theoretical_argmax = [62.7, 60.1]
    theoretical_sample_from = [54.1, 53.6]

    lstm_argmax = [59.77,47.0]
    lstm_sample_from = [48.8,34.4]

    method = ("sample_from", "argmax")
    accuracy = {
        'Scratch Network': (scratch_sample_from[which], scratch_argmax[which]),
        'Keras Network (RNN, HIDDEN_DIM = 32)': (keras_32_sample_from[which], keras_32_argmax[which]),
        'Keras Network (RNN, HIDDEN_DIM = 128)': (keras_128_sample_from[which], keras_128_argmax[which]),
        'Keras Network (LSTM, HIDDEN_DIM = 32)': (lstm_sample_from[which], lstm_argmax[which]),
        'Theoretical Max': (theoretical_sample_from[which], theoretical_argmax[which]),
    }

    x = np.arange(len(method))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Comparison for {"Last" if which else "First"} Names')
    ax.set_xticks(x + 2*width, method)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 100)

    plt.show()

def generation_time_native():

    scratch = [5.8, 7.5]
    keras_32 = [4311,7204]
    keras_128 = [4116, 7519]
    lstm = [4007, 5834]

    which = ("First Names", "Last Names")
    accuracy = {
        'Scratch Network': (scratch[0], scratch[1]),
        'Keras Network (HIDDEN_DIM = 32)': (keras_32[0], keras_32[1]),
        'Keras Network (HIDDEN_DIM = 128)': (keras_128[0], keras_128[1]),
        'Keras Network (LSTM, HIDDEN_DIM = 32)': (lstm[0], lstm[1]),
    }

    x = np.arange(len(which))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        print(attribute)
        print(measurement)
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Generation Time (s)')
    ax.set_title(f'Generation Time Comparison for 10,000 Names')
    ax.set_xticks(x + 2*width, which)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, max(scratch + keras_32 + keras_128 + lstm)+500)

    plt.show()

def generation_time_eager_disabled():

    scratch = [5.8, 7.5]
    keras_32 = [267,373]
    keras_128 = [275, 374]
    lstm = [318, 434]

    which = ("First Names", "Last Names")
    accuracy = {
        'Scratch Network': (scratch[0], scratch[1]),
        'Keras Network (HIDDEN_DIM = 32)': (keras_32[0], keras_32[1]),
        'Keras Network (HIDDEN_DIM = 128)': (keras_128[0], keras_128[1]),
        'Keras Network (LSTM, HIDDEN_DIM = 32)': (lstm[0], lstm[1]),
    }

    x = np.arange(len(which))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Generation Time (s)')
    ax.set_title('Generation Time Comparison for 10,000 Names\n' +
                 'With Eager Execution Disabled')
    ax.set_xticks(x + 2*width, which)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, max(scratch + keras_32 + keras_128 + lstm)+50)

    plt.show()

def generation_time_all():

    scratch = [5.8, 7.5]
    graph = [4311,7204]
    eager = [267,373]
    tflite = [166, 177]

    which = ("First Names", "Last Names")
    accuracy = {
        'Scratch Network': (scratch[0], scratch[1]),
        'Keras - Eager Mode': (graph[0], graph[1]),
        'Keras - Graph mode': (eager[0], eager[1]),
        'Keras - TensorFlow Lite': (tflite[0], tflite[1]),
    }

    x = np.arange(len(which))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Generation Time (s)')
    ax.set_title('Generation Time Comparison for 10,000 Names\n' +
                 'Different Keras Modes')
    ax.set_xticks(x + 2*width, which)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, max(scratch + graph + eager + tflite)+500)

    plt.show()


def generation_accuracy():

    scratch = [63.1, 9.0]
    keras_32 = [72.7,14.54]
    keras_128 = [88.6,49.8]
    keras_lstm = [75.78, 21.26]

    which = ("First Names", "Last Names")
    accuracy = {
        'Scratch Network': (scratch[0], scratch[1]),
        'Keras Network (RNN, 32)': (keras_32[0], keras_32[1]),
        'Keras Network (RNN, 128)': (keras_128[0], keras_128[1]),
        'Keras Network (LSTM, 32)': (keras_lstm[0], keras_lstm[1]),
    }

    x = np.arange(len(which))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in accuracy.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% of Names Recreated')
    ax.set_title('Generation "Accuracy" Comparison\n')
    ax.set_xticks(x + 2*width, which)
    ax.legend(loc='upper right', ncols=1)
    ax.set_ylim(0, 100)

    plt.show()

def full_name_frequencies():
    namesfile =  "data/all_names.json"
    with open(namesfile,"r") as f:
        entries = json.load(f)
        players = [Player(entry) for entry in entries]

    full_names = []
    for i in range(len(players)):
        full_name = (players[i].firstname if players[i].firstname is not None else "")  + " " + players[i].lastname
        full_names.append(full_name)

    full_count = Counter(full_names)

    return full_count

    full_df = pd.DataFrame.from_dict(full_count,orient='index')
    full_df.sort_values(inplace=True)

    return full_df

#### Tests the frequency of generated first names vs the frequency
#### in the original data set, plotted by length of the name
    # duplicates = dictionary of generated names 
    # with keys 'firstnames' and 'lastnames', which are
    # generated by generation_test()
def generated_frequency_test(duplicates: dict):

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

def average_name_lengths():
    firstnames, lastnames, _ = import_names()
    fn_len = pd.Series([name[1:-1] for name in firstnames]).apply(len)
    ln_len = pd.Series([name[1:-1] for name in lastnames]).apply(len)

    which = ("Mean", "Median")
    length = {
        'First Names': (fn_len.mean(), fn_len.median()),
        'Last Names': (ln_len.mean(), ln_len.median()),
        'Company Names': (24.61624087591241, 23.0)
    }

    x = np.arange(len(which))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in length.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Length')
    ax.set_title('Mean and Median Length\n' +
                 'of Names in Training Data Sets')
    ax.set_xticks(x + width, which)
    ax.legend(loc='upper right', ncols=1)
    ax.set_ylim(0, 35)

    plt.show()
# generation_time_native()
# average_name_lengths()
# longtail_frequency()
# momentum_plot()
# learning_rate_plot()
# batch_size_plot()
# lastnames_comparison_plot()
# firstnames_comparison_plot()
# neurons_comparison_plot()
