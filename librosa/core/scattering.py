import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.signal
import matplotlib.pyplot as plt

def get_mother_frequency(bins_per_octave):

    """
    The dimensionless mother center frequency xi (corresponding to a log period
    \gamma=0) is computed as the midpoint between the center frequency of the second
    center frequency xi*2^(-1/bins_per_octave) (corresponding to \gamma=1) and the
    negative mother center frequency (1-xi). Hence the equation
    2 xi = xi*2^(-1/bins_per_octave) + (1-xi), from which we
    derive xi = 1 / (3 - 2^(1/bins_per_octave)). This formula is valid
    only when the wavelet is a symmetric bump in the Fourier domain.

    Function returns the dimensionless mother frequency

    Inputs
    ======
    bins_per_octave : Number of Filters per Octave

    Outputs
    =======
    mother_frequency : dimensionaless mother center frequency

    """

    mother_frequency = 1.0 / (3.0 - 2.0**(-1.0/bins_per_octave))
    return mother_frequency


def get_wavelet_filter_specs(bins_per_octave, qualityfactor, nOctaves):

    """
    Create wavelet filter specs : centerfrequecy, bandwidth at different gammas

    Inputs
    ======
    bins_per_octave
    qualityfactor
    nOctaves

    Outputs
    =======
    psi_specs[gamma] : gamma indexed dictionary that contains the (centerfrequency, bandwidth) tuple
                     : #gammas = bins_per_octave * nOctaves
    Need to to perform asserts before creating the filters (this)
    """

    mother_frequency = get_mother_frequency(bins_per_octave)
    psi_specs = {}
    for j in range(nOctaves):
        for q in range(bins_per_octave):
            gamma = j * bins_per_octave + q
            resolution = np.power(2,-gamma / bins_per_octave)
            centerfrequency = mother_frequency * resolution
            # unbounded_scale = h * max_qualityfactor / centerfrequency
            # scale = min(unbounded_scale, max_scale)
            # unbounded_q = scale * centerfrequency / h
            #clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
            # qualityfactor = clamp(unbounded_q, 1.0, max_qualityfactor)
            bandwidth = centerfrequency / qualityfactor
            psi_specs[gamma] = (centerfrequency, bandwidth)
    return psi_specs

def create_morlet_1d_bank(psi_specs, N, nOctaves,  bins_per_octave, qualityfactor):
    """
    Function returns morlet 1d wavelet filter bank and Littlewood Paley function

    Inputs
    ======
    N : length of input signal that is a power of 2 (after chunking)
    nOctaves : number of nOctaves log2(N) by default, if not any value smaller than this can be used
    bins_per_octave : number of wavelet filters per octave

    Outputs
    =======
    psi_filters : list of morlet wavelet filters for different gamma, with low pass filter
    lp : Littlewood payley function

    calculate bandpass filters \psi for size N and J different center frequencies and low pass filter \phi
    and resolutions res < log2(N)
    """
    #dictionary of wavelet filters
    psi = {}
    #full resolution
    N0 = N

    for gamma in psi_specs:
        #get the center frequency
        xi = psi_specs[gamma][0] #0.4 * 2 ** (-j)
        #get bandwidth : TODO add division by explicit qualityfactor here
        bandwidth = psi_specs[gamma][1]
        psi[gamma] = 2 * np.exp(- np.square(np.arange(0, N0, dtype=float) / N - xi) * 10 * np.log(2) / bandwidth ** 2).transpose()
        bw = xi/qualityfactor #0.4 * 2 ** (-1 + J)

    phi = np.exp(-np.square(np.arange(0, N0,dtype=float)) * 10 * np.log(2) / bw**2 ).transpose()

    # Calculate Littlewood-Paley function
    lp = np.zeros(shape=(N))
    for gamma in psi:
        lp = lp + 0.5 * np.square(np.abs(psi[gamma]))
    lp = lp + np.square(np.abs(phi[1]))
    temp = lp[0]
    lp = (lp + lp[::-1]) * .5
    lp[0] = temp
    lp = 0
    return (psi, phi, lp)

def calculate_convolution(sig, psi, phi, scattering_order):

    """
    Function computes scattering coefficientrs for order M over filters in psi and phi
    sig : input signal
    psi : wavelet filters dictionary
    phi : low pass filters dictionary
    scattering_order : scattering order (>= 1)
    TO BE COMPLETED
    """
    U = {}
    U[1] = sig
    S = {}
    # maximal length
    log2N = np.log2(len(psi[1]))
    # number of gammas : len(psi)
    for m in range(1, M + 2):
        lambda_idx = 1
        #create new dictionary : m+1 th order U signals and mth order S signals
        U[m + 1] = {}
        sigf = fft_module.fft(U[m])
        for s in range(1, len(U[m]) + 1):
            if m <= M:

                for j in range(s, len(psi) + 1):
                    U[m + 1][lambda_idx] = np.abs(fft_module.ifft(np.multiply(sigf, psi[j])))
                    lambda_idx = lambda_idx + 1

        S[m] = np.abs(fft_module.ifft(np.multiply(sigf, phi)))
    return (S, U)

def plot_filters(psi, phi, lp):
    """
    plot the constructed filters (in frequency domain)
    check for Littlewood Paley

    """

    plt.figure()
    for gamma in psi:
        plt.plot(psi[gamma],'r')
    plt.plot(phi,'b')
    plt.plot(lp,'k')
    return

#test scatteering
# create mother frequnecy and set of centerfrequency, bandwidth
# create wavelet filters
bins_per_octave = 1
qualityfactor = 1
nOctaves = 12
N = 2**16

assert(nOctaves < np.log2(N))

mother_frequency = get_mother_frequency(bins_per_octave)
psi_specs = get_wavelet_filter_specs(bins_per_octave, qualityfactor, nOctaves)
psi, phi, lp = create_morlet_1d_bank(psi_specs, N, nOctaves, bins_per_octave, qualityfactor)

plot_filters(psi, phi, lp)
