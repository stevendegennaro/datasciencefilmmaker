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

def plot_sync_problem(data):
    plt.scatter(data.iloc[0:100]['Julian Date'],np.repeat(1,100))

    date = data.iloc[0]['Julian Date']
    while date < data.iloc[100]['Julian Date']:
        plt.axvline(date,zorder=0,color="black")
        date += 179
        plt.show()

###########################
#### Fourier Transform ####
###########################

def julian_dates_to_x(julian_dates):
    return (julian_dates - min(julian_dates))/(1.00001*(max(julian_dates) - min(julian_dates))) - 0.5

def plot_nfft(x,y,y_r,k,f_k, cutoffs = None):

    if cutoffs:
        x = x[cutoffs[0]:cutoffs[1]]
        y = y[cutoffs[0]:cutoffs[1]]
        y_r = y_r[cutoffs[0]:cutoffs[1]]

    fig,ax = plt.subplots(3,1)

    ax[0].scatter(x,y,color='lightgray')
    ax[0].scatter(x,y_r.real,color='black',marker='.',s=1)

    ax[1].plot(x,y_r.real-y,color='red')

    ax[2].plot(k, f_k.real, label='real')
    ax[2].plot(k, f_k.imag, label='imag')
    ax[2].legend(loc="upper left")

    # fig.suptitle(f"N_k = {len(k)}")

    return ax



def get_nfft(julian_dates,intervals, cutoffs = None):

    julian_dates = np.array(julian_dates)
    intervals = np.array(intervals)
    x = julian_dates_to_x(julian_dates)
    x = x[(x >= -0.5) & (x < 0.5)]

    y = intervals - np.mean(intervals)
    N_k = len(x)
    if N_k % 2:
        N_k += 1
    k = -(N_k // 2) + np.arange(N_k)

    print(len(x),len(y))
    print("Calculating Fourier Transform")
    f_k = nfft.ndft_adjoint(x, y, len(k))

    print("Calculating Y values")
    y_r = nfft.ndft(x,f_k)/len(x)

    return x, y, y_r, k, f_k


def toy_model(N_x = 1000, 
              scatter_scale = 1.0, 
              n_components = 20, 
              random = False, 
              cutoffs = None, 
              interval_shift = True,
              zoom = True):
    average_interval = 179.0

    frequencies = []
    if random:
        intervals = average_interval + scatter_scale * (np.random.rand(N_x) - 0.5)
        zoom = False

    else:
        intervals = np.full(N_x,179.0)
        for i in range(n_components):
            if n_components == 1:
                n_cycles = 10
            else:
                n_cycles =  100 * np.random.rand()
            frequencies.append(n_cycles)
            # n_cycles = 10.0
            offset = np.random.rand() - 0.5
            new_component = scatter_scale * np.sin(n_cycles * 2.0 * np.pi * (np.arange(0,1,1/N_x) + offset))
            intervals += new_component


    if interval_shift:
        start = 1.0e6
        julian_dates = [start]
        for i in range(N_x - 1):
            julian_dates.append(julian_dates[i] + intervals[i])

    else:
        julian_dates = np.linspace(-0.5, 0.49999, num = len(intervals))

    data = import_solar_eclipse_data()
    julian_dates = data['Julian Date'].iloc[0:N_x]

    x, y, y_r, k, f_k = get_nfft(julian_dates,intervals)

    ax = plot_nfft(x, y, y_r, k, f_k, cutoffs)

    for frequency in frequencies:
        ax[2].axvline(frequency,color="black",zorder=0)
        ax[2].axvline(-frequency,color="black",zorder=0)

    if zoom:
        ax[2].set_xlim([-120,120])

    plt.show()

def intervals_nfft(data, 
                   use_only_major_intervals = True, 
                   x_range = None, 
                   equispaced = False,
                   cutoffs = None):

    if use_only_major_intervals:
        data = data[data['Time to Next'] > 170]

    julian_dates = np.array(data['Julian Date'])
    intervals = np.array(data['Time to Next'])

    if x_range:
        julian_dates = julian_dates[x_range[0]:x_range[1]]
        intervals = intervals[x_range[0]:x_range[1]]

    if equispaced:
        julian_dates = np.linspace(-0.5,0.4999,len(julian_dates))

    # N_x = len(julian_dates)
    # # # average_interval = 179.0
    # # # intervals = average_interval + 1.0 * (np.random.rand(N_x) - 0.5)

    # jd_x = julian_dates_to_x(julian_dates)
    # intervals = np.sin(10 * 2.0 * np.pi * jd_x)

    x, y, y_r, k, f_k = get_nfft(julian_dates,intervals)
    # fig,ax = plt.subplots(1,1)

    # ax.scatter(x,y)
    # ax.scatter(x,y_r.real,color='black',marker='.',s=1)

    plot_nfft(x, y, y_r, k, f_k, cutoffs = cutoffs)


def plot_fft(x,y,y_r,f_k, cutoffs = None):

    period = (x - min(x))/365.2425

    if cutoffs:
        x = x[cutoffs[0]:cutoffs[1]]
        y = y[cutoffs[0]:cutoffs[1]]
        y_r = y_r[cutoffs[0]:cutoffs[1]]

    fig,ax = plt.subplots(3,1)
    ax[0].scatter(x,y,color="lightgray")

    ax[2].plot(period,f_k.real, label='real')
    ax[2].plot(period,f_k.imag, label='imag')

    ax[1].plot(x,y_r.real-y,color='red')

    ax[0].scatter(x,y_r,color="black",marker='.',s=1)

def intervals_fft(data, use_only_major_intervals = False,cutoffs = None):
    if use_only_major_intervals:
        data[data['Time to Next'] < 170] = np.nan
        data['Time to Next'] = data['Time to Next'].interpolate()
    y = np.array(data['Time to Next'] - data['Time to Next'].mean())
    x = np.array(range(len(y)))
    # y = np.cos((10 * 2 * np.pi * x)/len(x))

    print("Calculating transform")
    f_k = np.fft.fft(y)
    print("Calculating reverse transform")
    y_r = np.fft.ifft(f_k)

    x = data.iloc[x]['Julian Date']

    plot_fft(x,y,y_r,f_k, cutoffs = cutoffs)

    plt.suptitle(f"Equispaced Disctrete Fourier Transform of {'Major' if use_only_major_intervals else 'All'} Eclipse Intervals")
    plt.show()



def nfft_test(N_x = 10000, 
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
    N_k = len(x)
    if N_k % 2:
        N_k += 1
    k = -(N_k/2) + np.arange(N_k)
    print("Calculating Fourier Transform")
    f_k = nfft.ndft_adjoint(x, y, len(k))

    print("Calculating Y values")
    y_r = nfft.ndft(x,f_k)/len(x)

    plt.ion()
    fig,ax = plt.subplots(3,1)

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
# https://dsp.stackexchange.com/questions/101/how-to-extrapolate-a-1d-signal
# https://www.tradingview.com/script/u0r2gpti-Fourier-Extrapolator-of-Price-w-Projection-Forecast-Loxx/
# https://gist.github.com/MCRE-BE/f40daf732886d091b0886e071abf9e75


# Environment Installs:
# matplotlib
# pandas
# scipy
# jdcal
# ipython

