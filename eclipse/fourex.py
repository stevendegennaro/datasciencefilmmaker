import numpy as np
import matplotlib.pyplot as plt

def fourierExtrapolation(f_k, n_predict, harm_fraction = 0.5):
    
    # Get the indexes for the n_harm frequencies
    # with the highest amplitudes
    harm_fraction = max(min(harm_fraction,1.0),0.0)
    n_harm = int(len(f_k) * harm_fraction)
    biggest_f_ks = np.argsort(np.absolute(f_k))[-n_harm:]
    new_f_k = np.zeros(len(f_k),dtype=complex)
    new_f_k[biggest_f_ks] = f_k[biggest_f_ks]

    # reverse the fft
    # y_r = np.fft.ifft(new_f_k,len(y))

    N = len(f_k)
    n = np.tile(np.arange(N + n_predict), (N, 1)).transpose()
    k = np.array(np.arange(N))
    y_r = (new_f_k * np.exp(2 * np.pi * np.sqrt(-1+0j) * (np.arange(0,N) * n / N)) / N).sum(axis=1)

    return y_r