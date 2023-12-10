import pandas as pd
import matplotlib.pyplot as plt

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
    title = f"Scratch Network Trained on First Names\n"
    ax.set_title(title)
    plt.show()
tf_batch_size_plot()