import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from name_network import import_names
from collections import Counter

def learning_rate_plot():
    losses1 = pd.read_csv("weights/firstnames_0.01_500_history.txt")
    losses2 = pd.read_csv("weights/firstnames_0.05_500_history.txt")
    losses3 = pd.read_csv("weights/firstnames_0.005_500_history.txt")

    fig,ax = plt.subplots()
    ax.plot(losses1['Time'],losses1["Accuracy"],color="red", label="Learning Rate = 0.01")
    ax.plot(losses2['Time'],losses2["Accuracy"],color="black", label="Learning Rate = 0.05")
    ax.plot(losses3['Time'],losses3["Accuracy"],color="blue",label="Learning Rate = 0.005")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

def batch_size_plot():
    losses1 = pd.read_csv("weights/firstnames_0.01_500_history.txt")
    losses2 = pd.read_csv("weights/firstnames_0.01_1000_history.txt")
    losses3 = pd.read_csv("weights/firstnames_0.01_18187_history.txt")

    fig,ax = plt.subplots()
    ax.plot(losses1['Time'],losses1["Accuracy"],color="red", label="Batch Size = 500")
    ax.plot(losses2['Time'],losses2["Accuracy"],color="black", label="Batch Size = 1,000")
    ax.plot(losses3['Time'],losses3["Accuracy"],color="blue",label="Batch Size = 18,187")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

batch_size_plot()

def momentum_plot():
    losses1 = pd.read_csv("weights/firstnames_0.01_1000_history.txt")
    losses2 = pd.read_csv("weights/firstnames_0.01_1000_0.5_history.txt")
    losses3 = pd.read_csv("weights/firstnames_0.01_1000_0.1_history.txt")
 
    fig,ax = plt.subplots()
    ax.plot(losses1['Time'],losses1["Accuracy"],color="red", label="Momentum = 0.9")
    ax.plot(losses2['Time'],losses2["Accuracy"],color="black", label="Momentum = 0.5")
    ax.plot(losses3['Time'],losses3["Accuracy"],color="blue",label="Momentum = 0.1")
    ax.set_xlabel("Time in Seconds")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()

# momentum_plot()
# learning_rate_plot()
# batch_size_plot()

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
    print(len(ln_df),len(fn_df))
    n_names = 200
    ln_df = ln_df.iloc[0:n_names]
    fn_df = fn_df.iloc[0:n_names]
    # print(ln_df)
    # print(fn_df)
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



# longtail_frequency()
