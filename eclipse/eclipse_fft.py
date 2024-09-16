import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jdcal
import nfft
from import_eclipse_data import import_solar_eclipse_data
from fourex import fourierExtrapolation
import scipy.optimize
from collections import Counter
from matplotlib.widgets import Slider, Button

########################
#### NDFT Transform ####
########################

def convert_x_for_ndft(input_x: list[float]) -> list[float]:
    '''
    The NDFT algorithm requires x values that range from [-0.5,0.5)
    This function takes a list of x values and converts them so that
    they cover that range
    '''
    return (input_x - min(input_x))/(1.00001*(max(input_x) - min(input_x))) - 0.5

def plot_ndft(x: np.array, 
              y: np.array,
              y_r: np.array,
              k: np.array,
              f_k: np.array,
              set_xlimits: tuple[int,int] = None,
              top_panel_only = False) -> plt.Axes:
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

    if top_panel_only:
        fig,ax = plt.subplots()
        ax = [ax]
    else:
        fig,ax = plt.subplots(3,1)

    ax[0].scatter(x,y)#,color='lightgray')
    ax[0].scatter(x,y_r.real,color='black',marker='.')#,s=1)

    if not top_panel_only:
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


def get_ndft_frequencies(tt: np.array, yy: np.array):
             # intervals: np.array) -> tuple[np.array,np.array,np.array,np.array,np.array]:
    ''' Calculates the non-equispaced Fouier transform of a
        series of data, given a list of dates and the intervals between them
        (could be used for any date, though)
        It then takes the x values and the fourier coefficients
        and recreates the intervals. Process is not exact, so this 
        will in general not be the same as the original values
    '''
    
    x = convert_x_for_ndft(tt)
    x = x[(x >= -0.5) & (x < 0.5)]
    y = yy - np.mean(yy)
    N_k = len(x)
    if N_k % 2:
        N_k += 1
    k = -(N_k // 2) + np.arange(N_k)

    print("Calculating Fourier Transform")
    f_k = nfft.ndft_adjoint(x, y, len(k))

    print("Calculating Inverse Fourier Transform")
    # y_r = nfft.ndft(x,f_k)/len(x)

    return x, y, y_r, k, f_k

##########################
#### Create Sinusoids ####
##########################

def sinfunc(x, A, w, p, c, x0 = 0):
    return A * np.sin(w * (x - x0) + p) + c

def sines_from_df(x, df, index = None):
    '''
    Calculates the value of a set of sinusoids at the
    given x values from a dataframe containing the
    sinusoid components, where each row is one sinusoid
    with amplitude 'a', angular frequency 'w', phase 'p'
    and offset 'c'

    If you give it an index, it uses only that row number. 
    If you give it no index, it adds all of the components.
    '''
    if index != None:
        rows = [index, index + 1]
    else:
        rows = [0,len(df)]
    y_hat = np.zeros(len(x))
    for i in range(rows[0],rows[1]):
        y_hat += sinfunc(x, df.iloc[i]['a'], df.iloc[i]['w'], df.iloc[i]['p'], df.iloc[i]['c'])
    return y_hat

def create_sinusoids(x, 
                     mean_y = 179.0, 
                     amplitude_scale: float = 1.0, 
                     period_scale: float = 10.0,
                     n_components: int = 20,
                     noise_amplitude: float = 0.0):

    N_x = len(x)

    omegas = np.zeros(n_components)
    amplitudes = np.zeros(n_components)
    phases = np.zeros(n_components)
    offsets = np.zeros(n_components)

    x_range = max(x) - min(x)
    for i in range(n_components):
        if n_components == 1:
            n_cycles = period_scale
            amplitude = amplitude_scale
            phase = 0.0
        else:
            n_cycles =  np.abs(np.random.normal(0,period_scale))
            # Nothing higher than nyquist to avoid aliasing
            while (2 * n_cycles > N_x):
                n_cycles =  np.abs(np.random.normal(0,period_scale))
            amplitude = max(0,np.random.normal(amplitude_scale,0.5 * amplitude_scale))#/np.sqrt(n_components)
            phase = 2 * np.pi * np.random.rand()
        omega = 2 * np.pi * n_cycles / x_range 
        omegas[i] = omega
        amplitudes[i] = amplitude
        phases[i] = phase
        offsets[i] = 0.0

    parameters = pd.DataFrame({'w':omegas, 'a':amplitudes, 'p':phases, 'c':offsets})
    parameters.sort_values('a',ascending = False,inplace = True)
    parameters.reset_index(inplace=True, drop=True)
    parameters.loc[0,'c'] = parameters.loc[0,'c'] + mean_y

    y_noise = sines_from_df(x,parameters) + np.random.normal(0,noise_amplitude,len(x))

    return y_noise, parameters


def intervals_ndft(data: pd.DataFrame, 
                   use_only_major_intervals: bool = True, 
                   use_x_range: tuple[int,int] = None, 
                   equispaced: bool = False,
                   set_xlimits: tuple[int,int] = None,
                   top_panel_only: bool = False) -> None:

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
    jd_x = julian_dates_to_x(julian_dates)
    intervals = np.sin(10 * 2.0 * np.pi * jd_x)

    x, y, y_r, k, f_k = get_ndft(julian_dates,intervals)

    ax = plot_ndft(x, y, y_r, k, f_k, set_xlimits = set_xlimits,top_panel_only = top_panel_only)

    plt.show()

    return x, y, y_r, k, f_k


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
    y, _ = create_sinusoids(1000,179.0,1.0,n_frequences)#N_x = 1000, mean_y = 179.0, amplitude_scale = 1.0, n_components = 20
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


def plot_repeat():
    x = np.arange(0,100,1)
    y,df = create_sinusoids(x)
    plt.plot(x,y,'.k')
    x = np.arange(100,200,1)
    plt.plot(x,y,'.r')
    x = np.arange(0,99,.01)
    y = sines_from_df(x,df)
    plt.plot(x,y,'b-',zorder=0)
    x = np.arange(100,199,.01)
    plt.plot(x,y,'b:',zorder=0)
    plt.plot([99,100],[y[-1],y[1]],'b:')
    plt.show()

def fit_sin(tt, yy, uniform = True):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))

    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess,method='lm')
    A, w, p, c = popt

    return {"a": A, "w": w, "p": p, "c": c}

def fit_sinusoidal_components(x, y, n_to_fit):

    mean_y = y.mean()
    residuals = y - mean_y

    # Fit and plot the sine waves
    components = pd.DataFrame(columns=["a","w","p","c"])
    for i in range(0,n_to_fit):
        components.loc[len(components.index)] = fit_sin(x, residuals)
        residuals = residuals - sines_from_df(x,components,i)
    components.loc[0,"c"] = components.loc[0,"c"] + mean_y

    return components




def create_slider_plot(x_actual,y_actual):
     
    # x = np.linspace(0, 1, 1000)
    x = np.linspace(min(x_actual),max(x_actual),5*len(x_actual))
    
    # Defining the initial parameters
    init_amplitude = 1
    init_offset = 0.0
    max_frequency = np.log10(2*np.pi*len(x_actual)/(max(x_actual) - min(x_actual))/500)
    print(max_frequency)
    init_frequency = max_frequency
    init_phase = 0.0
    x0 = min(x)
     
    # Creating the figure and the graph line that we will update
    fig, ax = plt.subplots(figsize=(13,7))
    plt.plot(x_actual,y_actual,'.k')

    line, = plt.plot(x, sinfunc(x, init_amplitude, init_frequency, init_phase, init_offset, x0), lw=2)
    # ax.set_xlabel('Time [s]')
     
    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)
     
    # adjusting the main plot to make space for our sliders
    plt.subplots_adjust(left=0.2, bottom=0.2)
     
    # Making a horizontally oriented slider to 
    # control the frequency.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    freq_slider = Slider(
        ax=axfreq,
        label='w',
        valmin=-8,
        valmax=max_frequency,
        valinit=init_frequency,
        # orientation="horizontal" is Default
    )
    
    axphase = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    phase_slider = Slider(
        ax=axphase,
        label='p',
        valmin=0.0,
        valmax=2 * np.pi,
        valinit=init_phase,
        # orientation="horizontal" is Default
    )

    # Making a vertically oriented slider to control the amplitude
    axamp = plt.axes([0.05, 0.25, 0.0225, 0.63], facecolor=axcolor)
    amp_slider = Slider(
        ax=axamp,
        label="A",
        valmin=0,
        valmax=3,
        valinit=init_amplitude,
        orientation="vertical"
    )

    # Making a vertically oriented slider to control the amplitude
    axoff = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
    offset_slider = Slider(
        ax=axoff,
        label="c",
        valmin=-1.0,
        valmax=1.0,
        valinit=init_offset,
        orientation="vertical"
    )
     
    # Function to be rendered anytime a slider's value changes
    def update(val):
        line.set_ydata(sinfunc(x, amp_slider.val, 10**freq_slider.val, phase_slider.val, offset_slider.val,x0))
        fig.canvas.draw_idle()
     
    # Registering the update function with each slider Update
    freq_slider.on_changed(update)
    amp_slider.on_changed(update)
    offset_slider.on_changed(update)
    phase_slider.on_changed(update)
     
    plt.show()



def fit_sinusoidal_components_manual(x, y, n_to_fit):

    # mean_y = y.mean()
    # residuals = y - mean_y
    residuals = y

    # Fit and plot the sine waves
    components = pd.DataFrame(columns=["a","w","p","c"])
    for i in range(0,n_to_fit):
        create_slider_plot(x,residuals)
        components.loc[len(components.index)] = fit_sin(x, residuals)
        residuals = residuals - sines_from_df(x,components,i)
    # components.loc[0,"c"] = components.loc[0,"c"] + mean_y

    return components

def plot_fits(x,
              y_actual,
              p_fit, 
              p_actual = None,
              plot_components = False,

              n_fits = 6):

    y_fit = sines_from_df(x,p_fit)

    if plot_components:
        # Create axes
        max_plots = min([n_fits,len(p_fit)])
        fig, axes = plt.subplots(max_plots + 1)


        residuals = y_fit
        for i in range(0,max_plots + 1):

            # Plot the current y values being fit
            axes[i].scatter(x,residuals,color="lightgray")

            # Plot the actual sinusoids that were used to generate
            # the data (if it is fake data)
            if p_actual is not None:
                if i < len(p_actual):
                    axes[i].plot(x,sines_from_df(x,p_actual,i),'b-', linewidth=3)

            # If it's not the final panel (which is just the 
            # remaining residual of the entire fit)
            if i < len(p_fit): 
                # Plot the fit of the current component
                axes[i].plot(x,sines_from_df(x,p_fit,i), "r--", linewidth=2)

                # Calculate the next component y values
                residuals = residuals - sines_from_df(x,p_fit,i)

    # Create new figure of final fit
    fig,axes = plt.subplots(2)
    # Extend x range to twice as long
    # so we can see how it is extrapolating
    x_range = np.linspace(min(x),max(x),5*len(x))

    if p_actual is not None: axes[0].plot(x_range, sines_from_df(x_range,p_actual), "b-")
    axes[0].plot(x_range,sines_from_df(x_range,p_fit),'r--',zorder=0)
    axes[0].scatter(x, y_actual, color = "lightgray")
    axes[0].scatter(x, y_fit,color = 'black',marker='.')
    axes[1].scatter(x, y_actual - y_fit,color = 'black',s=3)
    if p_actual is not None: axes[1].plot(x_range,sines_from_df(x_range,p_actual) - sines_from_df(x_range,p_fit))

    return fig, axes

def fit_test(N_x = 1000, 
             n_components = 2,
             period_scale = 50.0,
             n_to_fit = None,
             noise_amplitude:float = 0.0,
             seed = None):

    if seed is None: 
        seed = np.random.randint(100)
        print(seed)
    np.random.seed(seed)

    if not n_to_fit: n_to_fit = n_components + 1

    x = np.arange(N_x)
    x = np.linspace(991085,2460408,N_x)
    y_actual, p_actual = create_sinusoids(x = x, mean_y = 179.0, amplitude_scale = 1.0, \
                                     period_scale = period_scale, n_components = n_components, \
                                     noise_amplitude = noise_amplitude)
    p_fit = fit_sinusoidal_components(x,y_actual,n_to_fit)
    plot_fits(x, y_actual, p_fit, p_actual, plot_components = True)

    plt.show()


def toy_model(n_components = 2):
    N_x = 1000
    x = np.arange(N_x)
    x = np.linspace(991085,2460408,N_x)
    x_range = max(x) - min(x)
    n_cycles = 10

    components = []
    phases = 2*np.pi * np.random.random(n_components)
    amplitudes = 1.0 + 0.1 * np.random.random(n_components) - 0.5
    for i in range(n_components):
        component = sines_from_df(x,pd.DataFrame({'w':[2 * np.pi * n_cycles / x_range], 'a':[amplitudes[i]], 'p':[phases[i]], 'c':[179.0]}))
        components.append(component[i::n_components])
    # c2 = sines_from_df(x,pd.DataFrame({'w':[2 * np.pi * n_cycles / x_range], 'a':[1.0], 'p':[np.pi], 'c':[179.0]}))

    y_actual = np.ravel(components,'F')

    # plt.plot(x,y_actual,'.k')
    # plt.show()
    # return
    p_fit = fit_sinusoidal_components(x,y_actual,n_components + 1)
    # return p_fit
    plot_fits(x, y_actual, p_fit, plot_components = False)
    plt.show()



def predict_variance(n_data_points_to_use = 1000, n_predict = 100, n_frequencies = 2):
    data = import_solar_eclipse_data()
    y_actual = np.array(data['Variance'])[9561 - n_data_points_to_use:9561 + n_predict]
    x = np.arange(len(y_actual))
    x = np.array(data['Julian Date'])[9561 - n_data_points_to_use:9561 + n_predict]


    x_train = x[:n_data_points_to_use]
    y_train = y_actual[:n_data_points_to_use]
    x_test = x[-n_predict:]
    y_test = y_actual[-n_predict:]

    p_fit = fit_sinusoidal_components_manual(x_train,y_train,n_frequencies)
    # fig, axes = plot_fits(x, y_actual, p_fit, plot_components = True)
    # axes[0].axvline(n_data_points_to_use)
    # plt.show()


# def toy_model(N_x: int = 1000,
#               amplitude_scale: float = 1.0,
#               random: bool = False,
#               n_components:int = 20,  
#               use_real_eclipse_dates = False,
#               interval_shift: bool = True,
#               set_xlimits:tuple[int,int] = None,
#               plot = False) -> None:
#     ''' Toy model for eclipses  
#         Creates random sinusoids or random uniform data
#         Runs NDFT on the toy model and then tries to recover
#         the original y values. Plots the results, residuals, 
#         and power spectrum

#         N_x = the number of x,y pairs it will create
#         amplitude_scale = amplitude of sinusoids or +-y limit of random data
#         random = use uniform random (use sinusoids if false)
#         n_components = number of sinusoids to create
#         use_real_eclipse_dates = take dates from the data itself
#         interval_shift = use the intervals between each
#             eclipse to determine the x values. If false, 
#             x values are equispaced on [-0.5,0.5)
#         zoom_f_k_graph = zoom in on the bottom graph to show detail
#         set_xlimits = high and low values of x,y to throw away before plotting
#             (still used in the analysis)
#     '''
#     average_interval = 179.0 # average time between eclipses

#     frequencies = []
#     if random:
#         intervals = average_interval + amplitude_scale * (np.random.rand(N_x) - 0.5)
#         zoom_f_k_graph = False

#     else:
#         intervals,frequencies = create_sinusoids(N_x, average_interval, amplitude_scale, n_components)
#         zoom_f_k_graph = True

#     if interval_shift:
#         start = 1.0e6
#         julian_dates = [start]
#         for i in range(N_x - 1):
#             julian_dates.append(julian_dates[i] + intervals[i])

#     else:
#         julian_dates = np.linspace(-0.5, 0.49999, num = len(intervals))

#     if use_real_eclipse_dates:
#         data = import_solar_eclipse_data()
#         julian_dates = data['Julian Date'].iloc[0:N_x]

#     x, y, y_r, k, f_k = get_ndft(julian_dates,intervals)

#     if plot:
#         ax = plot_ndft(x, y, y_r, k, f_k, set_xlimits)

#         for frequency in frequencies:
#             ax[2].axvline(frequency,color="black",zorder=0)
#             ax[2].axvline(-frequency,color="black",zorder=0)

#         if zoom_f_k_graph:
#             ax[2].set_xlim([-120,120])

#         plt.show()

# def predict_next_eclipses(data, n_data_points_to_use = 1000, n_predict = 100, n_frequencies = 10):
#     data = data.copy()
#     data = data[data['Time to Next'] > 170]
#     # y_actual = np.array(data[(data['Year']<=2023) | ((data['Year'] == 2024) & (data['Month'] == 4))]['Time to Next'])
#     y_actual = np.array(data['Time to Next'].head(n_data_points_to_use))
#     x = np.arange(n_data_points_to_use)
#     p_fit, residuals = fit_sinusoidal_components(x,y_actual,n_frequencies)
#     y_extend = np.array(data['Time to Next'].head(n_data_points_to_use + n_predict))
#     x_extend = np.arange(n_data_points_to_use + n_predict)

#     # fig, axes = plot_fits(x, y_actual, p_fit, residuals)
#     fig, axes = plot_fits(x_extend, y_extend, p_fit, residuals)
#     axes[0].axvline(n_data_points_to_use,color='lightblue',linestyle=':')
#     axes[1].axvline(n_data_points_to_use,color='lightblue',linestyle=':')
#     # axes[0].scatter(x_extend,y_extend,color = "lightblue")
#     # axes[0].scatter(x_extend, sines_from_df(x,p_fit),color = 'black',marker='.',s=2)
#     # axes[1].scatter(x, y_extend - sines_from_df(x_extend,p_fit),color = 'darkblue',s=3)
#     plt.show()






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
