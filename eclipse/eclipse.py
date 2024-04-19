import pandas as pd
import calendar
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import jdcal
import scipy
import nfft
import sys

def convert_time_to_julian_date_fraction(time_string):
    h,m,s = time_string.split(':')
    return (60 * 60 * int(h) + 60 * int(m) + int(s))/(24*60*60)

def calculate_julian_dates(data):
    julian_date_list = []
    for index, row in data.iterrows():
        if row['Year'] < 1582 or (row['Year'] == 1582 and row['Month'] < 10):
            julian_date = jdcal.jcal2jd(row['Year'],row['Month'],row['Day'])
        else:
            julian_date = jdcal.gcal2jd(row['Year'],row['Month'],row['Day'])
        julian_date = julian_date[0]-0.5 + julian_date[1]
        julian_date += convert_time_to_julian_date_fraction(row['Time'])
        julian_date_list.append(julian_date)
    return julian_date_list

def import_solar_eclipse_data():
    filename = 'data/eclipse.gsfc.nasa.gov_5MCSE_5MCSEcatalog_no_header.txt'

    data = pd.read_fwf(filename,header=None,infer_nrows=10000)
    data.set_index(0,inplace=True)
    data.index.name = 'Cat. Number'
    data.columns = ('Canon Plate','Year','Month','Day','Time',\
                    'DT','Luna Num','Saros Num','Type','Gamma','Ecl. Mag.',\
                    'Lat.','Long.','Sun Alt','Sun Azm','Path Width','Central Duration')

    #### Fix dates ###
    # Convert month abbreviations to numbers
    month_abbr = [m for m in calendar.month_abbr]
    data['Month'] = data['Month'].map(lambda m: month_abbr.index(m)).astype('Int8')
    # Convert to Julian Dates
    data['Julian Date'] = calculate_julian_dates(data)

    # Calculate intervals between eclipses
    data['Time to Next'] = pd.DataFrame(data['Julian Date'].diff()).shift(-1)
    data = data[:-1]

    return data


############################
#### Plotting Functions ####
############################

def plot_interval_histogram(data):
    plt.hist(data['Time to Next'],bins =  100)
    plt.title('Histogram of Intervals Between Solar Eclipses')
    plt.xlabel('Interval')
    plt.ylabel('Frequency')
    plt.show()


def plot_interval_histogram_zoomed(data):
    intervals = data.loc[data['Time to Next'] > 170,'Time to Next']
    plt.hist(intervals,bins =  100)
    plt.title('Histogram of Intervals Between Solar Eclipses')
    plt.xlabel('Interval')
    plt.ylabel('Frequency')
    plt.show()

def plot_interval_histogram_short_zoomed(data):
    intervals = data.loc[data['Time to Next'] < 100,'Time to Next']
    plt.hist(intervals,bins =  100)
    plt.title('Histogram of Intervals Between Solar Eclipses')
    plt.xlabel('Interval')
    plt.ylabel('Frequency')
    plt.show()


def plot_interval_histogram_total(data):
    total_eclipses = data[(data['Type'] == 'T') | (data['Type'] == 'H')]
    intervals = pd.DataFrame(total_eclipses['Julian Date'].diff())
    intervals = intervals[intervals > 170]
    plt.hist(intervals,bins =  100)
    plt.title('Histogram of Intervals Between Total/Hybrid Solar Eclipses')
    plt.xlabel('Interval')
    plt.ylabel('Frequency')
    plt.show()

def plot_interval_vs_time_major(data):
    # intervals = data.loc[data['Time to Next'] > 170,'Time to Next']
    long_intervals = data['Time to Next'] > 170
    plt.scatter(data[long_intervals]['Julian Date'],
                data[long_intervals]['Time to Next'],
                marker = '.',
                color='black'
                )

    # start = data.loc[1,'Julian Date']
    # stop = data.iloc[-1]['Julian Date']
    # saros = np.arange(start,stop,6585.3211)
    # for xc in saros:
    #     plt.axvline(x=xc,color='blue')

    plt.title('Major Intervals Between Solar Eclipses')
    plt.ylabel('Interval')
    plt.xlabel('Julian Date')
    plt.show()

def plot_interval_vs_time(data, plot_saros = True, xlim = (1.464e6,1.51e6)):
    # intervals = data.loc[data['Time to Next'] > 170,'Time to Next']
    # long_intervals = data['Time to Next'] > 170
    plt.scatter(data['Julian Date'],
                data['Time to Next'],
                marker = '.',
                s = 1,
                color='black'
                )

    if plot_saros:
        start = data.loc[1,'Julian Date']
        stop = data.iloc[-1]['Julian Date']
        saros = np.arange(start,stop,6585.3211)
        for xc in saros:
            plt.axvline(x=xc,color='blue')

    plt.title('Intervals Between Solar Eclipses')
    plt.ylabel('Interval')
    plt.xlabel('Julian Date')
    if xlim:
        plt.xlim(xlim)
    plt.show()

    # print(lines[0])

###########################
#### Fourier Transform ####
###########################


def time_to_next(data, jd_list):
    index_list = (data['Julian Date'].searchsorted(jd_list)).clip(0,len(data)-1)
    difference = data.iloc[index_list]['Julian Date'] - jd_list
    return difference 


def intervals_fft(data):
    x = np.arange(int(np.round(data.iloc[0]['Julian Date'])),int(np.round(data.iloc[-1]['Julian Date'])))
    y = time_to_next(data,x)
    plt.scatter(x,y,marker='.',s=1)
    plt.show()

    # f = scipy.interpolate.interp1d(data['Julian Date'], data['Time to Next'])
    # return f
    # x = np.arange(data.iloc[0]['Julian Date'],data.iloc[-1]['Julian Date'],1/24.0/60.0)
    # y = f(x)

    # plt.ion()
    # plot_interval_vs_time(data, plot_saros = False, xlim = (1.464e6,1.51e6))
    # plt.scatter(x,y,marker='.',s=1,color='red')
    # plt.ioff()

def julian_dates_to_x(julian_dates):
    return (julian_dates - min(julian_dates))/(1.00001*(max(julian_dates) - min(julian_dates))) - 0.5

def x_to_jd():
    pass

def my_ifft(f_k):
    n = len(f_k)  # Length of the input array
    # time_indices = np.arange(n)  # Time indices
    A = []
    for m in range(n):
        a_m = 0
        for k in range(n):
            # print(m,k,a_m,f_k * np.exp(2 * np.pi * 1j * (m * k / n)))
            a_m += f_k[k] * np.exp(2 * np.pi * 1j * (m * k / n))
            print(m,k,a_m)
        a_m /= n
        A.append(a_m)
        # print(A)

    # x = (1/N) * np.sum(f_k * np.exp(1j * 2 * np.pi * np.outer(n, np.arange(N)) / N), axis=1)
    return A

def toy_model():
    average_interval = 179.0
    interval_dispersion = 1.0
    N = 1000
    cycles = 10
    intervals = interval_dispersion * np.cos(cycles * 2 * np.pi * np.arange(0,N) / N) + average_interval

    start = 1.0e6
    julian_dates = [start]
    for i in range(N-1):
        julian_dates.append(julian_dates[i] + intervals[i])

    julian_dates = np.array(julian_dates)
    intervals = np.array(intervals)
    # intervals -= np.mean(intervals)
    x = julian_dates_to_x(julian_dates)
    f = intervals - np.mean(intervals)
    n_k = 500
    k = -(n_k // 2) + np.arange(n_k)
    f_k = nfft.ndft_adjoint(x, f, len(k))

    # return(f_k)

    # f_reconstructed = np.fft.ifft(f_k)

    fig,ax = plt.subplots(2,1)

    ax[0].scatter(x,f,marker='.')
    ax[1].plot(k, f_k.real, label='real')
    ax[1].plot(k, f_k.imag, label='imag')
    ax[1].legend()
    # 
    # plt.scatter(x,f_x_direct,marker='.')
    # print(f_k)
    # return f_k

    plt.show()
    return f_k


def nfft_test(N_x = 10000, 
              N_k = 1000, 
              n_waves = 10, 
              scatter_scale = 0.01, 
              sin = True, 
              equal_space = False):
    x = np.arange(-0.5,0.49999,1/N_x)
    if not equal_space:
        x += np.random.normal(scale = scatter_scale, size = N_x)
        x = x[(x >= -0.5) & (x < 0.5)]
        x = np.sort(x)

    y = np.sin(n_waves * 2 * np.pi * x) if sin else np.cos(n_waves * 2 * np.pi * x)
    # print(N_k, N_k % 2)
    if N_k % 2:
        N_k += 1

    k = -(N_k/2) + np.arange(N_k)
    print("Calculating Fourier Transform")
    f_k = nfft.ndft_adjoint(x, y, len(k))

    plt.ion()
    fig,ax = plt.subplots(3,1)

    print("Calculating Y values")
    y_r = nfft.ndft(x,f_k)/len(x)

    ax[0].scatter(x,y)
    ax[0].scatter(x,y_r.real,color='black',marker='.',s=1)

    ax[1].plot(x,y_r.real-y,color='red')

    ax[2].plot(k, f_k.real, label='real')
    ax[2].plot(k, f_k.imag, label='imag')
    ax[2].legend()

    fig.suptitle(f"N_x = {N_x}, N_k = {N_k}")

    plt.show()

 #                           TD of
 # Cat. Canon    Calendar   Greatest          Luna Saros Ecl.           Ecl.                Sun  Sun  Path Central
 #  No. Plate      Date      Eclipse     DT    Num  Num  Type  Gamma    Mag.   Lat.   Long. Alt  Azm Width   Dur.


 # Imported data from https://eclipse.gsfc.nasa.gov/SEpubs/5MCSE.html
 # "FIVE MILLENNIUM CANON OF SOLAR ECLIPSES: -1999 TO +3000"

 # datetime objects can't handle BC

 # Average time between solar eclipses = 153.5 days
 # Average time between total eclipses = 575.6 days = 18.9 months
 # Average time between total and hybrid = 488.1 = 16 months


# In [422]: data[(data['Time to Next']>170) & (data['Time to Next']<175.9)]
# Out[422]: 
#              Canon Plate  Year  Month  Day  ... Central Duration            Date String   Julian Date  Time to Next
# Cat. Number                                 ...                                                                    
# 490                   25 -1801     10    9  ...           00m58s  -01801-10-09T18:49:02  1.063550e+06    175.677118
# 3085                 155  -700      2   23  ...           03m16s  -00700-02-23T06:12:09  1.465454e+06    175.094722
# 4059                 203  -300      2   11  ...                -  -00300-02-11T00:14:24  1.611539e+06    175.486597
# 4521                 227  -101     11   22  ...           00m03s  -00101-11-22T23:06:25  1.684506e+06    175.479931
# 5257                 263   199     10    7  ...           00m40s  +00199-10-07T19:06:10  1.794033e+06    175.703796
# 7861                 394  1300      2   21  ...           01m24s  +01300-02-21T08:34:00  2.195937e+06    175.099595

# "There is some historical uncertainty as to which years from 43 BCE to 8 CE were counted as leap years. For the purposes of this web site, we assume that all Julian years divisible by 4 are be counted as leap years."

# https://medium.com/geekculture/down-the-rabbit-hole-of-event-prediction-a-guide-to-time-related-event-analysis-and-beyond-7529591adada
# https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/
# https://github.com/daynebatten/keras-wtte-rnn
# https://github.com/ragulpr/wtte-rnn/tree/master/python
# https://pypi.org/project/pynufft/0.3.2.8/
# https://dsp.stackexchange.com/questions/16590/non-uniform-fft-dft-with-fftw
# https://github.com/jakevdp/nfft


# Environment Installs:
# matplotlib
# pandas
# scipy
# jdcal
# ipython

