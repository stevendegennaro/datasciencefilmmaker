import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from import_eclipse_data import import_solar_eclipse_data
from collections import Counter


def interval_runs_histogram(interval_runs):

    count = Counter(interval_runs)
    count_df = pd.DataFrame.from_dict(count, orient='index').astype(np.int32)
    count_df.index = pd.MultiIndex.from_tuples(count_df.index)
    count_df.sort_index(inplace=True)
    count_df.columns = ['Count']
    count_df.unstack().plot(kind='bar')
    plt.show()


def tokenize_time_to_next(time_to_next):
    time_to_next[time_to_next > 170] = 200
    time_to_next[(time_to_next < 170) & (time_to_next > 140)] = 100
    time_to_next[time_to_next < 99] = 0
    time_to_next /= 100
    time_to_next = list(time_to_next.astype(np.int32))

    return time_to_next

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

def 
