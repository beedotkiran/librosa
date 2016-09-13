import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.fftpack as fft_module

"""
Scattering transform specs : 

Paramters for the scattering transform function : 
    creating wavelet filters : 
    nOctaves, 
    bins_per_octave, 
    quality factor

    calculating scattering : 
    order 
    wavelet_filter for each order 
    averaging window T (fixed over all orders) 


The specifications psi_specs shall contain a dictionary index by a tuple of 
valid gammas as shown below and whose values shall be a tuple containing
the filters center frequency and bandwidth.

psi_spec[(\gamma_1, \gamma_2, .. \gamma_M)] = (centerfrequency_M, bandwidth_M)
"""

def get_mother_frequency(bins_per_octave):

    """
    The dimensionless mother center frequency xi (corresponding to a log period
    \gamma=0) is computed as the midpoint between the center frequency of the 
    second center frequency xi*2^(-1/bins_per_octave) (corresponding to \gamma
    equals 1) and the negative mother center frequency (1-xi). Hence the eqn.
    2 xi = xi*2^(-1/bins_per_octave) + (1-xi), from which we derive : 
    xi = 1 / (3 - 2^(1/bins_per_octave)). This formula is valid only when the 
    wavelet is a symmetric bump in the Fourier domain.

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

def get_wavelet_filter_specs(bins_per_octave, quality_factor, nOctaves, scattering_order):

    """
    Create wavelet filter specs : centerfrequecy, bandwidth at different gammas
    Wavelet filter specs are independent of the signal length and resolution.

    Inputs
    ======
    bins_per_octave (scalar or list of size scattering_order)
    quality_factor (scalar or list of size scattering_order)
    nOctaves (scalar or list of size scattering order)
    scattering_order

    Outputs
    =======
    For scattering_order = 1
    psi_specs[gamma] : gamma indexed dictionary that contains the (centerfrequency, bandwidth) tuple
                     : #gammas = bins_per_octave * nOctaves
    
    TO BE DONE :
    For scattering_order = 2
    psi_specs[(gamma_1, gamma_2)] = (centerfrequency, bandwidth)
    where (gamma_1 ,gamma_2) corresponds to paths in the scattering transform
    This generalizes thus for any scattering_order > 1 in the same way.
    To be sure that we calculate valid wavelet filters in higher orders we check the following condition :
    
    xi_1 2^(-\gamma_1/Q1) / Q1 > xi_2 2^(-\gamma_2/Q2)
    ----------------------------------------------------
    Bandwidth of the wavelet filter (@some gamma) for order M  > Center frequency for 
    wavelet filter(@gamma) for order M+1.

    at Q1=Q1=NFO=1 (dyadic) we have xi_1 = xi_2
    we have \gamma_1 < \gamma_2
    
    Need to to perform asserts before creating the filters (this)
    
    """

    mother_frequency = get_mother_frequency(bins_per_octave)
    psi_specs = {}
    for j in range(nOctaves):
        for q in range(bins_per_octave):
            gamma = j * bins_per_octave + q
            resolution = np.power(2,-gamma / bins_per_octave)
            centerfrequency = mother_frequency * resolution
            # unbounded_scale = h * max_quality_factor / centerfrequency
            # scale = min(unbounded_scale, max_scale)
            # unbounded_q = scale * centerfrequency / h
            #clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
            # quality_factor = clamp(unbounded_q, 1.0, max_quality_factor)
            bandwidth = centerfrequency / quality_factor
            psi_specs[gamma] = (centerfrequency, bandwidth)
    return psi_specs

def create_morlet_1d_bank(psi_specs, N, nOctaves,  bins_per_octave):
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
    TODO : correct the bandwidth of low pass filter
    TODO : Renormalize using Littlewood payley factor
    TODO : Morlet with corrections 
    calculate bandpass filters \psi for size N and J different center frequencies and low pass filter \phi
    and resolutions res < log2(N)
    """
    #dictionary of wavelet filters
    psi = {}
    f = np.arange(0, N, dtype=float) / N #normalized frequency axis
    FWHM_factor = 10*np.log(2)
    for gamma in psi_specs:
        print('Filter' + repr(gamma))
        #get the center frequency
        fc, bandwidth_psi = psi_specs[gamma]
        #create filters at full resolution of input signal
        #fc is the normalized center frequency 
        den = bandwidth_psi**2/FWHM_factor
#        psi[gamma] = _morlet_1d(fc,den,N,nPeriods=5)
        psi[gamma] = 2 * np.exp(- np.square(f - fc)  / den).transpose()
    
    phi = np.exp(-np.square(np.arange(0, N,dtype=float)) * 10 * np.log(2) / bandwidth_psi**2 ).transpose()
    
    
    #The max and min of LP sum needs to be close to 1 (bounded LP sum approximately preserves norm)
    lp = np.zeros(shape=(N))
    for gamma in psi:
        lp = lp + 0.5*np.square(np.abs(psi[gamma]))
    lp = lp + np.square(np.abs(phi[1]))
    lp[1::] = (lp[1::] + lp[-1:0:-1])*0.5
    normalizer_lp = np.max(lp)
    print(normalizer_lp)

    return (psi, phi, lp)

def _gauss(x, den):
    """ gaussian function centered at x and with sigma**2 set to den """
    
    return np.exp(-np.square(x)/den)

def _morlet_1d(center_freq, den, N, nPeriods=5):
    """ Function creates morlet wavelet at length 5N and corrects the magnitude 
    at -2N, -N, 0, N, 2N by resolving a linear system 
    den : denominator of the gaussian (bw**2/(2*np.log(2)))    
    y : morlet 1d wavelet of length N
    """
    
    """ Computer range of frequencies with non-neglible magnitude """    
    halfN = N>>1
    pstart = -((nPeriods-1)>>1)
    pstop = (nPeriods-1)>>1 + np.remainder(nPeriods+1,2) #iseven
    omega_start = -halfN + pstart*N
    omega_stop = halfN + pstop*N-1

    """ compute gaussians """
    gauss_center = [_gauss(omega-center_freq,den) for omega in 
                   np.arange(omega_start, omega_stop+1, dtype=float)]
    gauss_zeros = [_gauss(omega,den) for omega in 
                  np.arange(omega_start+pstart*N, omega_stop + pstop*N, dtype=float)]
    corrective_gaussians = [gauss_zeros[omega+p*N] for omega, p in 
                           zip(np.arange(N*nPeriods),np.arange(nPeriods))] 
    
    """ Computer corrective weights """
    b = [_gauss(p*N-center_freq,den) for p in np.arange(pstart,pstop+1)]
    A = np.zeros((nPeriods,nPeriods)) #check size
    for p in np.arange(nPeriods):
        for q in np.arange(nPeriods):
            A[p,q] = _gauss((q-p)*N,den)

    corrective_factors = np.linalg.solve(A, b)
    
    """ Periodize in Fourier and average """ 
    y = gauss_center - np.dot(corrective_gaussians,corrective_factors)
    y = y.reshape((N,nPeriods))
    y = np.sum(y, axis=1) #no squeeze required
    
    return y
    
def scattering_main(input_signal, scattering_order, psi, phi, bins_per_octave, nOctaves):
    	
    """
    Function computes scattering coefficientrs for order M over filters in psi and phi
    
    for all orders
        for all signals of previous order:
            decompose (wavelet transform, modulus, subsampling)
            smooth (low pass, subsampling)
        end
    end
    """
    # get filters (intially construct with cauchy)
    # need to also add correction for morlet

    U = {}
    U[1] = {}
    S = {}
    S[1] = {}
    # for the scale and resolution information
    U[1][1] = {}
    
	# Initialize using input S[scattering_order][gamma]
	#it would be better to have a S
    U[1][1]['signal']=input_signal
    U[1][1]['scale']=-1
    U[1][1]['resolution']=0
    for m in range(1,scattering_order+1):
        raster=1
		
        for s in U[m]:
            sig=U[m][s]['signal']

			# Decompose/propagate - retrieve high frequencies
            if m<=scattering_order:
                children=decompose_signal(sig,U[m][s]['resolution'],U[m][s]['scale'],psi)
                for ic in children:
                    U[m+1][raster]=children[ic]
                    raster=raster+1

            # Smoothing the filtered signals
            S[m][s]=smooth_signal(sig,U[m][s]['resolution'],U[m][s]['scale'],psi,phi)
    return U, S
    
def decompose_signal(sig,resolution,scale,psi):
	""" Decompose a signal into its wavelet coefficients, downsample in fourier and compute the modulus"""
	
	number_of_j = len(psi[1])
	sigf=fft_module.fft(sig)
	prev_j=(scale>=0)*np.mod(scale,number_of_j) + -100*(scale<0)
	raster=1
	children={}
	for j in range(max(0,next_bands(prev_j)),len(psi[resolution+1])):
         children[raster] = {}
         children[raster]['signal'] = np.abs(calculate_convolution(sigf,psi[resolution+1][j+1]), resolution)
         children[raster]['resolution'] = resolution
         children[raster]['scale'] = (scale>=0)*scale*number_of_j + j
         raster=raster+1

	return children	

def smooth_signal(sig,resolution,scale,psi,phi):
    
    """Smooth a signal and downsample"""
	
    sigf=fft_module.fft(sig)
    number_of_j = len(psi[1])
    smoothed_signal = {}
    smoothed_signal['resolution']=resolution
    smoothed_signal['scale']=scale
    smoothed_signal['signal'] = calculate_convolution(sigf,phi[resolution+1])
    smoothed_signal['norm'] = np.linalg.norm(sig.ravel())

    return smoothed_signal

def calculate_convolution(sigf, filter_f, resolution):
    """ calculate convolution by mutliplying fft coefts
    Calculate the final resolution based on filter_f bandwidth and center freq.
    """
    
    
    
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
    plt.title('Littlewood Paley function' + 'bins/octave=' + repr(bins_per_octave) + ',Q-factor=' + repr(quality_factor))
    plt.show()
    return
   
#test scatteering
# create mother frequnecy and set of centerfrequency, bandwidth
# create wavelet filters
bins_per_octave = 1
quality_factor = 1
nOctaves = 12
N = 2**16
scattering_order = 1

#load test file from librosa/tests/data/test1_22050.wav
(sr, y) = scipy.io.wavfile.read('test1_22050.wav')
# this is a stereofile so averaging the two channels
y = y.mean(axis=1)
leny = len(y)
N = int(np.power(2, np.floor(np.log2(leny)) + 1))
input_signal = np.zeros(shape=(N))
input_signal[0:leny] = y


assert(nOctaves < np.log2(N))

mother_frequency = get_mother_frequency(bins_per_octave)

psi_specs = get_wavelet_filter_specs(bins_per_octave, quality_factor, nOctaves,scattering_order)

psi, phi, lp = create_morlet_1d_bank(psi_specs, N, nOctaves, bins_per_octave)

plot_filters(psi, phi, lp)

#S, U = scattering_main(input_signal, scattering_order, psi, phi, bins_per_octave, nOctaves)