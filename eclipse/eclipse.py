import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jdcal
import nfft
from import_eclipse_data import import_solar_eclipse_data
from fourex import fourierExtrapolation
import scipy.optimize

########################
#### NDFT Transform ####
########################

def julian_dates_to_x(julian_dates: list[float]) -> list[float]:
    '''
    The NDFT algorithm requires x values that range from [-0.5,0.5)
    This function takes a list of Julian dates and converts them so that
    they cover that range
    '''
    return (julian_dates - min(julian_dates))/(1.00001*(max(julian_dates) - min(julian_dates))) - 0.5

def plot_ndft(x: np.array, 
              y: np.array,
              y_r: np.array,
              k: np.array,
              f_k: np.array,
              set_xlimits: tuple[int,int] = None) -> plt.Axes:
    '''
    Function for plotting a 3-panel graph of the NDFT of a set of data
    x = the x values
    y - y values
    y_r = the values that are recreated when running the inverse fourier
    k = frequencies
    f_k = fourier coefficients
    set_xlimits = high and low values of x to throw away
    '''

    if set_xlimits:
        x = x[set_xlimits[0]:set_xlimits[1]]
        y = y[set_xlimits[0]:set_xlimits[1]]
        y_r = y_r[set_xlimits[0]:set_xlimits[1]]

    fig,ax = plt.subplots(3,1)

    ax[0].scatter(x,y,color='lightgray')
    ax[0].scatter(x,y_r.real,color='black',marker='.',s=1)

    ax[1].plot(x,y_r.real-y,color='red')

    ax[2].plot(k, f_k.real, label='real')
    ax[2].plot(k, f_k.imag, label='imag')
    ax[2].legend(loc="upper left")

    return ax

def get_ndft(julian_dates: np.array,
             intervals: np.array) -> tuple[np.array,np.array,np.array,np.array,np.array]:
    ''' Calculates the non-equispaced Fouier transform of a
        series of data, given a list of dates and the intervals between them
        (could be used for any date, though)
        It then takes the x values and the fourier coefficients
        and recreates the intervals. Process is not exact, so this 
        will in general not be the same as the original values
    '''
    
    julian_dates = np.array(julian_dates)
    intervals = np.array(intervals)
    x = julian_dates_to_x(julian_dates)
    x = x[(x >= -0.5) & (x < 0.5)]

    y = intervals - np.mean(intervals)
    N_k = len(x)
    if N_k % 2:
        N_k += 1
    k = -(N_k // 2) + np.arange(N_k)

    print("Calculating Fourier Transform")
    f_k = nfft.ndft_adjoint(x, y, len(k))

    print("Calculating Inverse Fourier Transform")
    y_r = nfft.ndft(x,f_k)/len(x)

    return x, y, y_r, k, f_k

def create_sinusoids(x, 
                     mean_y = 179.0, 
                     y_scale: float = 1.0, 
                     x_scale: float = 10.0,
                     n_components:int = 20) -> tuple[float,float]:

    omegas = np.zeros(n_components)
    amplitudes = np.zeros(n_components)
    phases = np.zeros(n_components)

    y = np.full(len(x),mean_y)
    x_range = max(x) - min(x)
    for i in range(n_components):
        if n_components == 1:
            n_cycles = x_scale
            amplitude = y_scale
            phase = 0.0
        else:
            n_cycles =  np.random.normal(0,x_scale)
            amplitude = max(0,np.random.normal(y_scale,0.5 * y_scale))
            phase = 2 * np.pi * np.random.rand()
        omega = 2 * np.pi * n_cycles / x_range 
        omegas[i] = omega
        amplitudes[i] = amplitude
        phases[i] = phase
        new_component = amplitude * np.sin(omega * x + phase)
        y += new_component

    parameters = pd.DataFrame({'w':omegas, 'a':amplitudes, 'p':phases})
    parameters.sort_values('a',ascending = False,inplace = True)

    return y, parameters


def toy_model(N_x: int = 1000,
              y_scale: float = 1.0,
              random: bool = False,
              n_components:int = 20,  
              use_real_eclipse_dates = False,
              interval_shift: bool = True,
              set_xlimits:tuple[int,int] = None,
              plot = False) -> None:
    ''' Toy model for eclipses  
        Creates random sinusoids or random uniform data
        Runs NDFT on the toy model and then tries to recover
        the original y values. Plots the results, residuals, 
        and power spectrum

        N_x = the number of x,y pairs it will create
        y_scale = amplitude of sinusoids or +-y limit of random data
        random = use uniform random (use sinusoids if false)
        n_components = number of sinusoids to create
        use_real_eclipse_dates = take dates from the data itself
        interval_shift = use the intervals between each
            eclipse to determine the x values. If false, 
            x values are equispaced on [-0.5,0.5)
        zoom_f_k_graph = zoom in on the bottom graph to show detail
        set_xlimits = high and low values of x,y to throw away before plotting
            (still used in the analysis)
    '''
    average_interval = 179.0 # average time between eclipses

    frequencies = []
    if random:
        intervals = average_interval + y_scale * (np.random.rand(N_x) - 0.5)
        zoom_f_k_graph = False

    else:
        intervals,frequencies = create_sinusoids(N_x, average_interval, y_scale, n_components)
        zoom_f_k_graph = True

    if interval_shift:
        start = 1.0e6
        julian_dates = [start]
        for i in range(N_x - 1):
            julian_dates.append(julian_dates[i] + intervals[i])

    else:
        julian_dates = np.linspace(-0.5, 0.49999, num = len(intervals))

    if use_real_eclipse_dates:
        data = import_solar_eclipse_data()
        julian_dates = data['Julian Date'].iloc[0:N_x]

    x, y, y_r, k, f_k = get_ndft(julian_dates,intervals)

    if plot:
        ax = plot_ndft(x, y, y_r, k, f_k, set_xlimits)

        for frequency in frequencies:
            ax[2].axvline(frequency,color="black",zorder=0)
            ax[2].axvline(-frequency,color="black",zorder=0)

        if zoom_f_k_graph:
            ax[2].set_xlim([-120,120])

        plt.show()


def intervals_ndft(data: pd.DataFrame, 
                   use_only_major_intervals: bool = True, 
                   use_x_range: tuple[int,int] = None, 
                   equispaced: bool = False,
                   set_xlimits: tuple[int,int] = None) -> None:

    ''' Runs real eclipse data through the ndft.
        use_only_major_intervals = only intervals > 170 days
        use_x_range = specifices the range of eclipses to fit
        equispaced = treat the data as equispaced
            if false, use actual Julian dates for each eclipse as x
        set_xlimits = high and low indices of x,y to throw away before plotting
            (still used in the analysis)
    '''

    if use_only_major_intervals:
        data = data[data['Time to Next'] > 170]

    julian_dates = np.array(data['Julian Date'])
    intervals = np.array(data['Time to Next'])

    if use_x_range:
        julian_dates = julian_dates[use_x_range[0]:use_x_range[1]]
        intervals = intervals[use_x_range[0]:use_x_range[1]]

    if equispaced:
        julian_dates = np.linspace(-0.5,0.4999,len(julian_dates))

    ### To test random data at real Julian Dates
    # N_x = len(julian_dates)
    # average_interval = 179.0
    # intervals = average_interval + 1.0 * (np.random.rand(N_x) - 0.5)

    ### To test a sine wave at real Julian dates
    # jd_x = julian_dates_to_x(julian_dates)
    # intervals = np.sin(10 * 2.0 * np.pi * jd_x)

    x, y, y_r, k, f_k = get_ndft(julian_dates,intervals)

    ax = plot_ndft(x, y, y_r, k, f_k, set_xlimits = set_xlimits)

    plt.show()


def plot_rfft(x: np.array,
              y: np.array,
              y_r: np.array,
              f_k: np.array,
              plot_by_freq: bool = False,
              set_xlimits = None) -> None:
    ''' Similar to plot_ndft but for fast fourier transforms '''

    # to plot period as number of years
    print(len(f_k))
    period = (max(x) - min(x))/(np.arange(len(f_k)))/364.2425
    # freq = (np.arange(len(f_k)))/(max(x) - min(x))
    # period = 1/freq/364.2425
    # period = np.insert(period,0,0)

    if set_xlimits:
        x = x[set_xlimits[0]:set_xlimits[1]]
        y = y[set_xlimits[0]:set_xlimits[1]]
        y_r = y_r[set_xlimits[0]:set_xlimits[1]]

    x = (x - min(x))/365.2524 - 2024

    fig,ax = plt.subplots(3,1)
    ax[0].scatter(x,y,color="lightgray")
    ax[0].scatter(x,y_r,color="black",marker='.',s=1)
    ax[1].plot(x,y_r.real-y,color='red')
    if plot_by_freq: ax[2].plot(np.arange(len(f_k)),np.absolute(f_k),color='darkorange')
    else: ax[2].plot(period,np.absolute(f_k),color='darkorange')
    ax[2].set_xscale('log')

    return ax

def intervals_rfft(data: pd.DataFrame,
                   no_future: bool = False,
                   use_only_major_intervals: bool = False, 
                   set_xlimits: tuple[int,int] = None, 
                   plot_by_freq: bool = False) -> None:
    if use_only_major_intervals:
        data = data.copy()
        data[data['Time to Next'] < 170] = np.nan
        data['Time to Next'] = data['Time to Next'].interpolate()
    
    if no_future:
        data = data[(data['Year']<=2023) | ((data['Year'] == 2024) & (data['Month'] == 4))]

    y = np.array(data['Time to Next']) #- data['Time to Next'].mean())
    x = np.array(range(len(y)))
    # y = np.sin((57 * 2 * np.pi * x)/len(x)) + 5
    y = y - y.mean()

    print("Calculating transform")
    f_k = np.fft.rfft(y)
    print("Calculating reverse transform")
    y_r = np.fft.irfft(f_k,len(x))

    x = data.iloc[x]['Julian Date']

    ax = plot_rfft(x,y,y_r,f_k, set_xlimits = set_xlimits, plot_by_freq = plot_by_freq)

    ax[2].axvline(6585.321/365.2425)

    plt.suptitle(f"Equispaced Disctrete Fourier Transform of {'Major' if use_only_major_intervals else 'All'} Eclipse Intervals")
    plt.show()
 
#####################
### Extrapolation ###
#####################

def extrapolate_and_get_difference(f_k, y, n_predict, harm_fractions):
    fit = fourierExtrapolation(f_k, n_predict, harm_fractions)
    difference = fit.real - y
    fit_difference = sum(difference[:-n_predict] ** 2)
    ex_difference = sum(difference[-n_predict:] ** 2)
    return fit, difference, fit_difference, ex_difference

def predict_next(data, n_frequences = 10, harm_fractions = 0.5):
    # np.random.seed(0)
    y = np.array(data['Time to Next']) #- data['Time to Next'].mean())
    y, _ = create_sinusoids(1000,179.0,1.0,n_frequences)#N_x = 1000, mean_y = 179.0, y_scale = 1.0, n_components = 20
        # 1000, 179, 10, 1)
    # y = np.array(data[(data['Year']<=2023) | ((data['Year'] == 2024) & (data['Month'] == 4))]['Time to Next'])
    n_predict = int(0.3 * len(y))
    f_k = np.fft.fft(y[:-n_predict])

    fd = []
    ed = []
    h = np.arange(0,1,0.01)
    for harm_fractions in h:
        fit, _, fit_difference, ex_difference = extrapolate_and_get_difference(f_k, y, n_predict, harm_fractions)
        fd.append(fit_difference)
        ed.append(ex_difference)

    fig,ax = plt.subplots(2,1)

    ax[1].plot(h,ed)
    best_fit = np.argmin(ed)
    print(best_fit, h[best_fit], ed[best_fit])
    # plt.plot(h,fd)

    fit, _, fit_difference, ex_difference = extrapolate_and_get_difference(f_k, y, n_predict, h[best_fit])


    # extrapolation = fit[-n_predict:]
    ax[0].plot(y, 'black', label = 'actual', linewidth = 1)
    ax[0].plot(fit[:-n_predict], 'r', linestyle = "--",label = 'fit')
    ax[0].plot(np.arange(len(f_k), len(f_k) + n_predict), fit[-n_predict:], \
        'b', linestyle = ":",label = 'extrapolation')

    plt.show()


#############################
#### Miscellaneous Plots ####
#############################

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

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess,method='trf')
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def fit_and_plot_components(x,y,n_to_fit,axes):

    # Fit and plot the sine waves
    components = []
    for i in range(0,n_to_fit + 1):
        if i == 0:
            new_y = y
        else:
            new_y = new_y - components[i-1]["fitfunc"](x)
        axes[i].scatter(x, new_y, color="lightgray")
        if i < n_to_fit: 
            components.append(fit_sin(x, new_y))
            axes[i].plot(x, components[i]["fitfunc"](x), "r--", linewidth=2)

    y_hat = components[0]["fitfunc"](x)
    for i in range(1,len(components)):
        y_hat += components[i]["fitfunc"](x)

    # Create new figure of final fit
    fig,ax = plt.subplots()
    ax.scatter(x, y, color = "lightgray")
    ax.plot(x,y_hat,'r--')

    return fig,ax

def fit_test(N_x = 1000, 
             n_components = 2,
             x_scale = 10.0,
             n_to_fit = None,
             noise_amplitude:float = 0.0,
             seed = None):

    if not seed: 
        seed = np.random.randint(100)
        print(seed)
    np.random.seed(seed)

    if not n_to_fit: n_to_fit = n_components + 1

    x = np.arange(N_x)
    y, parameters = create_sinusoids(x = x, mean_y = 0.0, y_scale = 1.0, \
                                     x_scale = x_scale, n_components = n_components)
    y_noise = y + np.random.normal(0,noise_amplitude,len(y))


    # Create axes
    fig, axes = plt.subplots(n_to_fit + 1)
    # Plot each actual components on top axis
    for i in range(n_components):
        axes[0].plot(x,parameters['a'][i] * np.sin(parameters['w'][i] * x + parameters['p'][i]),'b-', linewidth=3)

    fig, ax = fit_and_plot_components(x,y_noise,n_to_fit,axes)

    ax.plot(x, y, "b-")

    plt.show()





# def pyestimate_test(N_x = 1000, 
#              n_components = 2,
#              n_to_fit = None,
#              noise_amplitude:float = 0.0,
#              seed = None):

#     if seed: np.random.seed(seed)
#     if not n_to_fit: n_to_fit = n_components

#     x = np.arange(N_x)
#     y, parameters = create_sinusoids(x, 0.0, 1.0, n_components)
#     y_noise = y + np.random.normal(0,noise_amplitude,len(y))

#     # fit
#     A, f, phi = pyestimate.multiple_sin_param_estimate(y_noise, 5)

#     # reconstruct signal
#     y_hat = 0
#     for i in range(len(A)):
#         y_hat += A[i] * np.cos(2*np.pi*f[i]*x+phi[i])

#     # plot result
#     plt.plot(y, '-', label='original signal')
#     plt.plot(y_noise, '.', label='noisy input data')
#     plt.plot(y_hat, 'r--', label='fitted signal')
#     plt.legend()
#     plt.show()

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
# https://medium.com/thedeephub/mastering-time-series-forecasting-revealing-the-power-of-fourier-terms-in-arima-d34a762be1ce#:~:text=A%20Fourier%20series%20is%20an,%2C%20weekly%2C%20or%20monthly%20seasonality.
# https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
# https://dsp.stackexchange.com/questions/73911/is-there-an-a-method-to-fit-a-wave-created-from-two-wave

    # # Get the fourier frequencies
    # ff = 2.0 * np.pi * np.fft.rfftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    # Fy = abs(np.fft.rfft(y_noise))
    # fourier = pd.DataFrame({'ff':ff[1:],'Fy':Fy[1:]})
    # fourier = fourier.sort_values('Fy',ascending= False).reset_index(drop=True)


# def fit_sin_with_omega(tt, yy, w):
#     '''Fit sin to the input time sequence, and return fitting parameters "amp", "phase", "offset", "freq", "period" and "fitfunc"'''
#     tt = np.array(tt)
#     yy = np.array(yy)
#     guess_amp = np.std(yy) * 2.**0.5
#     guess_offset = np.mean(yy)
#     guess = np.array([guess_amp, 0., guess_offset])

#     def sinfunc(t, A, p, c):  return A * np.sin(w*t + p) + c
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, p, c = popt
#     f = w/(2.*np.pi)
#     fitfunc = lambda t: A * np.sin(w*t + p) + c
#     return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}



# Environment Installs:
# matplotlib
# pandas
# scipy
# jdcal
# ipython
# scikit-learn

