import pandas as pd
import matplotlib.pyplot as plt

def learning_rate_plot():
    lossesfile = "weights/firstnames_0.01_500_history.txt"
    losses1 = pd.read_csv(lossesfile)
    lossesfile = "weights/firstnames_0.05_500_history.txt"
    losses2 = pd.read_csv(lossesfile)
    lossesfile = "weights/firstnames_0.005_500_history.txt"
    losses3 = pd.read_csv(lossesfile)
    losses1 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses1.columns),losses1])
    losses2 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses2.columns),losses2])
    losses3 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses3.columns),losses3])

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
    lossesfile = "weights/firstnames_0.01_500_history.txt"
    losses1 = pd.read_csv(lossesfile)
    lossesfile = "weights/firstnames_0.01_1000_history.txt"
    losses2 = pd.read_csv(lossesfile)
    lossesfile = "weights/firstnames_0.01_18187_history.txt"
    losses3 = pd.read_csv(lossesfile)
    losses1 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses1.columns),losses1])
    losses2 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses2.columns),losses2])
    losses3 = pd.concat([pd.DataFrame([[0,0,0,0,""]], columns=losses3.columns),losses3])

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


# learning_rate_plot()
# batch_size_plot()

