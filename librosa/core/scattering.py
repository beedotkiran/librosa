import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io.wavfile
#import numpy.fft as fft_module
import scipy.fftpack as fft_module
from scipy.signal import chirp
import librosa


"""
TO DO 
- Check Dirac ouput : higher frequency coefficients still show low values.
- have to add separate filter bank for 2nd order Q (ususually Q2<Q1)

"""

def _ispow2(N):
    return 0 == (N & (N - 1))


def get_mother_frequency(bins_per_octave):
    """
    Function returns the dimensionless mother frequency

    Parameters
    ----------
    bins_per_octave : Number of Filters per Octave

    Returns
    -------
    mother_frequency : dimensionaless mother center frequency

    Notes
    -----

    The dimensionless mother center frequency xi (corresponding to a log period
    \gamma=0) is computed as the midpoint between the center frequency of the 
    second center frequency xi*2^(-1/bins_per_octave) (corresponding to \gamma
    equals 1) and the negative mother center frequency (1-xi). Hence the eqn.
    2 xi = xi*2^(-1/bins_per_octave) + (1-xi), from which we derive : 
    xi = 1 / (3 - 2^(1/bins_per_octave)). This formula is valid only when the 
    wavelet is a symmetric bump in the Fourier domain.

    References
    ----------
    .. [1] https://github.com/lostanlen/WaveletScattering.jl
           Implementation of scattering transform in Julia by V. Lostanlen
    """

    mother_frequency = 1.0 / (3.0 - 2.0**(-1.0 / bins_per_octave))
    return mother_frequency


def get_wavelet_filter_specs(bins_per_octave, quality_factor, nOctaves):
    """Create wavelet filter specs : centerfrequecy, bandwidth at different gammas
    Wavelet filter specs are independent of the signal length and resolution.

    Parameters
    ----------
    bins_per_octave : (scalar or list of size M)
        number of wavelets per octave in the fourier domain
    quality_factor : (scalar or list of size M)
        This is the ratio of center freq. to bandwidth
    nOctaves (scalar or list of size scattering order)
        number of octaves covering the fourier domain
    M : scattering order
        the order of scattering transform (max tested = 2)

    Returns
    -------

    psi_specs[gamma] : gamma indexed dictionary that contains the tuple 
                       (centerfrequency, bandwidth) 
                       #gammas = bins_per_octave * nOctaves

    Notes
    -----
    
    
    To be sure that we calculate valid wavelet filters in higher orders we need 
    to check the following condition :

    xi_1 2^(-\gamma_1/Q1) / Q1   >   xi_2 2^(-\gamma_2/Q2)

    Bandwidth of the wavelet filter (@some gamma) for order 
    M  > Center frequency for wavelet filter(@gamma) for order M+1.

    at Q1=Q2=bins_per_octave=1 (dyadic) we have xi_1 = xi_2
    we have the resulting condition to be :

    \gamma_1 < \gamma_2
    
    Though this does not hold for other values of Q1, Q2
    
    References
    ----------
    .. [1] https://github.com/lostanlen/WaveletScattering.jl
           Implementation of scattering transform in Julia by V. Lostanlen
           
    """

    mother_frequency = get_mother_frequency(bins_per_octave)
    psi_specs = {}
    fc_vec = []
    bw_vec = []
    
    for j in range(nOctaves):
        for q in range(0, bins_per_octave):
            gamma = j * bins_per_octave + q           
            resolution = np.power(2, -gamma / bins_per_octave)
            centerfrequency = mother_frequency * resolution
            bandwidth = centerfrequency / quality_factor
            psi_specs[(j,q)] = (centerfrequency, bandwidth)
            fc_vec.append(centerfrequency)
            bw_vec.append(bandwidth)
   
    if(display_flag):
        plt.figure()
        plt.plot(fc_vec)
        plt.plot(bw_vec)
        plt.title('Normalized bandwidth and center frequencies')
        plt.xlabel('Filter index')
        plt.ylabel('Normalized Center Frequency f in [0,1]')
        plt.legend(('fc','bw'))
        
    return psi_specs


def filterbank_morlet_1d(N, psi_specs, nOctaves):
    """Function returns complex morlet 1d wavelet filter bank 

    Parameters
    ----------
    N : integer > 0
        length of input signal that is a power of 2 (after chunking)
    nOctaves : integer > 0
        number of octaves/scales covering the frequency domain, 
        nOctaves <= log2(N) 
    bins_per_octave : integer > 0 
        number of wavelet filters per octave

    Returns
    -------
    psi : dictionary 
        dictionary of morlet wavelet filters indexed with different gamma 
    phi : array_like
        low pass filter
    lp : array_like 
        Littlewood payley function : Measure of quality of the filter bank for 
        signal representation, it should be as close as possible to 1.

    References
    ----------

    .. [1] Anden, J., Mallat S. 'Deep Scattering Spectrum'.  
          IEEE Transactions on Signal Processing 2014 

    Notes
    -----
    calculate bandpass filters \psi for size N and J different center 
    frequencies and low pass filter \phi and resolutions res < log2(N)
    
    The max. and min. of the littlewood payley function needs to be close to 1
    This preserves norm and ensures contractive operator

    TODO : correct the bandwidth of low pass filter (add this at the end)
    TODO : Morlet with corrections: _corrected_morlet_1d

    """

    psi = {} #wavelet
    lp = np.zeros(shape=(N)) #little-wood payley   
    lp_afternorm = np.zeros(shape=(N)) #little-wood payley
    FWHM_factor = 10 * np.log(2) #fullwidth at half max factor
#    psi_mat = np.zeros((N,len(psi_specs)))
    
    i = 0
    for index in psi_specs:
        fc, bandwidth_psi = psi_specs[index]
        den = bandwidth_psi**2 / FWHM_factor
        psi[index] = _morlet_1d(N, fc, den)
#        psi_mat[:,i] = psi[index]
        i = i + 1
        lp = lp + np.square(np.abs(psi[index]))
    
    f = np.arange(0, N, dtype=float) / N # normalized frequency domain        
    bandwidth_phi = 0.4 * 2**(-nOctaves+1)#is this the right bandwidth?    
    phi = np.exp(-np.square(f) * 10 * np.log(2) / bandwidth_phi**2)
    
#    plt.figure()
#    for index in psi_specs:
#        plt.plot(psi[index])
#    plt.plot(phi)    
#    plt.figure()
    

    lp = lp + np.square(np.abs(phi[1]))
    lp[1::] = (lp[1::] + lp[-1:0:-1]) * 0.5    
    
    normalizer_lp = np.max(np.sqrt(lp))
    
    for index in psi:
        psi[index] = psi[index] / normalizer_lp

    phi = phi / normalizer_lp

    for index in psi_specs:
        lp_afternorm = lp_afternorm + np.square(np.abs(psi[index]))

    lp_afternorm = lp_afternorm + np.square(np.abs(phi[1]))
    lp_afternorm[1::] = (lp_afternorm[1::] + lp_afternorm[-1:0:-1]) * 0.5
    
    if(0):        
        plt.figure()
        plt.title('Littlewood Payley function')
        plt.plot(lp)
        plt.plot(lp_afternorm)
        plt.legend(('Before Norm', 'After Norm'))
        
    filters = dict(phi=phi, psi=psi)

    return (filters, lp)

def filterbank_to_multiresolutionfilterbank(filters, max_resolution):
    """Converts a filter bank into a multiresolution filterbank
    For every filter in the filter bank, compute different resolutions 
    (differnt support). The input filters are assumed to be in the 
    This precalculated Multiresolution filter bank will speed up
    calculation of convolution of signal at the output of different
    wavelet filters and different resolutions.


    Parameters
    ----------
    filters : dictionary
        Set of filters stored in a dictionary in the following way:
         - filters['phi'] : Low pass filter at resolutions nOctaves to J
         - filters['psi'] : Band pass filter (Morlet) 
              where 'j' indexes the scale and 'q' indexes the bins_per_octave 
              of a single filter.

    max_resolution : int
        number of resolutions to compute for every filter 

    filterbank_to_multiresolutionfilterbank iterates first through 
    j in range(J) and then through the resolution based on current J. 
    This changes the structure of wavelet_filters
    
        for j in range(nOctaves):
            for q in range(bins_per_octave):
                for res in range(0,max(j-1,0)+1):
                    get_filter_at_resoluion(wavelet_filters['psi][(j,q)],res)                

    Returns
    -------
    filters_multires : dictionary
        Set of filters in the Fourier domain, at different scales and resolutions.
        See multiresolution_filter_bank_morlet1d for more details on the Filters_multires structure.

    """

    keys_jq = max(list(filters['psi'].keys()))
    nOctaves = keys_jq[0] + 1 #nOctaves
    bins_per_octave = keys_jq[1] + 1#bins_per_octave
    N = filters['psi'][(0,0)].shape[-1] #size at full resolution

    Phi_multires = []
    Psi_multires = {}

    for res in range(0,max_resolution):
        Phi_multires.append(_get_filter_at_resolution(filters['phi'],res))
    
    for j in range(nOctaves):
        for q in range(bins_per_octave):
            key = (j,q)
            Psi_multires[key] = {}
#            print(key)
            for res in range(0,max(j-1,0)+1):  
                Psi_multires[key][res] = _get_filter_at_resolution(filters['psi'][key],res)
#        print('')

    filters_multires = dict(phi=Phi_multires, psi=Psi_multires)
    
    return filters_multires
    
def _morlet_1d(N, fc, den):
    """Morlet wavelet at center frequency and bandwidth 
    
    Parameters
    ----------
    N : integer
        length of filter
    fc : float
        center frequency
    den : denominator of gaussian with sigma**2
    
    Returns
    -------
    Morlet filter with these parameters of length N
    
    """
    # normalized frequency axis
    f = np.arange(0, N, dtype=float) / N  
    return 2 * np.exp(- np.square(f - fc) / den).transpose()

def _apply_lowpass(x, phi, J, len_x_sub):
    """Applies a low pass filter on the signal x .
    Convolution by multiplying in fourier domain.
    Subsampling in time domain to obtain the correct resolution for given 
    number of scattering coeffs len_x_sub.

    Parameters
    ----------
    x : array_like
        Input signal in the spatial domain.
    phi : array_like
        Low pass filter in the Fourier domain.
    J : int
        Rate of subsampling 2**J
    n_scat : int
        number of spatial coefficients of the scattering vector

    Returns
    -------
    x_subsampled : ndarray
        input signal or stack of signals subsampled to the largest scale 
        determined by J
    """
    
    x = np.atleast_2d(x)
        
    x_sub = np.zeros((x.shape[0], len_x_sub))
    
    len_x = x.shape[1]
    ds = int(len_x/ len_x_sub)
    
    for q in range(x.shape[0]):
        xf = fft_module.fft(x[q,:])
        x_filtered = fft_module.ifft(xf*phi)
        x_sub[q,:] = 2 ** (J - 1) * x_filtered[::ds]
    
    return x_sub
        

def _get_filter_at_resolution(filt,j):
    """Computes filter 'filt' at resolution 'j'
    
    Parameters
    ----------
    filt : array
        Filter in the Fourier domain.
    j : int
        Resolution to be computed
    
    Returns
    -------
    filt_multires : array
    Filter 'filt' at the resolution j, in the Fourier domain
    
    """
    
#    cast = np.complex64
    N = filt.shape[0]  # filter is square
    
    assert _ispow2(N), 'Filter size must be an integer power of 2.'
    
    # Truncation in fourier domain and suming over responses from other bands
    # back into the truncated fourier domain (make sure there are no or 
    # neglible responses in these frequencies, otherwise leads to aliasing)
    mask = np.hstack((np.ones(int(N / 2 ** (1 + j))), 0.5, np.zeros(int(N - N / 2 ** (j + 1) - 1)))) \
        + np.hstack((np.zeros(int(N - N / 2 ** (j + 1))), 0.5, np.ones(int(N / 2 ** (1 + j) - 1))))
    
    #truncation by using mask and reshape
    filt_lp = np.complex64(filt*mask)
    
    # Remember: C contiguous, last index varies "fastest" (contiguous in
    # memory) (unlike Matlab)
    fold_size = (int(2 ** j), int(N / 2 ** j))
    filt_multires = filt_lp.reshape(fold_size).sum(axis=0)
    
    if debug_flag:
        plt.figure()
        plt.plot(mask)
        plt.plot(filt_lp)
        plt.plot(filt_multires)
        plt.title('resolution = ' + repr(j) + 'length of folded filter = ' + repr(len(filt_multires)))

    return filt_multires
    
def scattering(x,wavelet_filters=None,M=2):
    """Compute the scattering transform of a signal using the filter bank.

    
    Parameters
    ----------
    x : array_like
        input signal 
        Length of x needs to be a power of 2. 

    wavelet_filters : Dictionary 
        Multiresolution wavelet filter bank 
    M : int
        Order of the scattering transform, which can be 0, 1 or 2.


    Returns
    -------
    S : 2D array_like
        Scattering transform of the x signals, of size (num_coeffs, time). 

    U : array_like
        Result before applying the lowpass filter and subsampling.

    S_tree : dictionary
        Dictionary that allows to access the scattering coefficients (S) 
        according to the layer and indices. More specifically:
    
    Zero-order layer: The only available key is 0:
    S_tree[0] : 0th-order scattering transform
    S_tree[(j,q)] : 1st-order coefficients for scale 'j' and bin 'q'
    S_tree[((j1,q1),(j2,q2))] : 2nd-order for scale 'j1', bin 
    'q1' on the first layer, and 'j2', 'q2' in the second layer.

    The number of coefficients for each entry is (num_coefs,window_size)

    References
    ----------
    .. [1] Anden, J., Mallat S. 'Deep Scattering Spectrum'.  
            IEEE Transactions on Signal Processing 2014 
    .. [2] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'. 
            IEEE Transactions on PAMI, 2012.
    
    
    Examples
    --------

    """
    
    if(wavelet_filters==None):#build filters 
        N = len(x)        
        nOctaves = int(np.log2(len(x)))-1
        bins_per_octave = 2 
        quality_factor = 1
        
        psi_specs = get_wavelet_filter_specs(bins_per_octave, quality_factor, nOctaves)        
        filters, lp = filterbank_morlet_1d(N, psi_specs, nOctaves)
        wavelet_filters = filterbank_to_multiresolutionfilterbank(filters, nOctaves)
    
    if(debug_flag):
        #there are no truncated filters; checked at all resolutions 
        for key in wavelet_filters['psi']:
            plt.figure()
            plt.title('Key='+repr(key))
            for res in wavelet_filters['psi'][key]:
                plt.plot(wavelet_filters['psi'][key][res])
            
    keys_jq = max(list(wavelet_filters['psi'].keys()))
    J = keys_jq[0] + 1 
    Q = keys_jq[1] + 1
        
    num_coefs = {
        0: int(1),
        1: int(1 + J * Q),
        2: int(1 + J * Q + J * (J - 1) * Q**2 / 2)
    }.get(M, -1)

    window_size = int(x.shape[0]/2**(J-1)) #number of coeffecients in time
    
    if(print_flag):
        print('--> J = ' + repr(J) + ', Q = ' + repr(Q))
        print('# of Filters = J*Q' + repr(J*Q))
        print('# Scattering Coeffs='+repr(num_coefs))
        print('# Time window size = ' + repr(window_size))
    
    oversample = 1  # subsample at a rate a bit lower than the critic frequency

    U = []
    V = []
    v_resolution = []
    current_resolution = 0
    
    S = np.zeros((num_coefs,window_size)) #create matrix containing num_coeffs x window length 
    S_tree = {} #allows access to the coefficients (S) using the tree structure

    S[0, :] = _apply_lowpass(x, wavelet_filters['phi'][current_resolution], J,  window_size)[0] #first coefficient
    S_tree[0] = S[0, :]

    
    if M > 0: #First order scattering coeffs
        num_order1_coefs = J*Q
        S1 = S[1:J*Q+1,:].view()
        S1.shape=(num_order1_coefs,window_size)
        Xf = fft_module.fft(x) # precompute the fourier transform of the signal
        indx = 0
        filtmat = np.zeros((J*Q, len( wavelet_filters['psi'][(0,0)][current_resolution])))

        fig, axarr = plt.subplots(J, sharex=True)
        fig2, axarr2 = plt.subplots(J, sharex=True)
        for j in range(J):
            resolution = max(j-oversample, 0)
            v_resolution.append(resolution) # resolution for the next layer 
            for q in range(Q):
                filtersjq = wavelet_filters['psi'][(j,q)][current_resolution]
                #multiply in fourier and subsample in time
                ds =  2**resolution
                
                if(0): 
                    #when subsampling removed we have response at higher freq.
                    #need to be careful about ds. Sensitive.
                    print('Use no subsampling in lowpass filter nor in signal')
                    x_conv = fft_module.ifft(Xf*filtersjq) 
                    resolution = current_resolution
                else:
                    x_conv_f = Xf*filtersjq
                    mask = np.zeros(x_conv_f.shape)
                    len_x_conv_f = len(x_conv_f)
                    mask[:int(len_x_conv_f/ds)] = max(np.abs(x_conv_f))
                    axarr[indx].plot(np.abs(x_conv_f))
                    axarr[indx].plot(mask,'k--')
                    axarr[indx].set_title('Mask Length =' + repr(sum(mask>0)))
                    x_conv_f_truncate = x_conv_f[:int(len_x_conv_f/ds)]
#                    axarr2[indx].plot(np.abs(x_conv_f_truncate))
#                    axarr2[indx].set_title('Length =' + repr(len(x_conv_f_truncate)))

                    x_conv2 = fft_module.ifft(x_conv_f_truncate)
                    x_conv = ds*fft_module.ifft(Xf*filtersjq)[::ds]
                    diff = np.abs(x_conv2-x_conv)
                    axarr2[indx].plot(diff)
                    axarr2[indx].set_title(repr(np.mean(diff)))
                    print('--j, q, res = ' + repr((j,q,resolution)))

                V.append(x_conv)
                U.append(np.abs(x_conv))
                filtmat[indx, :] = np.abs(fft_module.ifft(Xf*filtersjq)) #np.abs(filtersj) 
                S1[indx, :] = _apply_lowpass(U[indx], wavelet_filters['phi'][resolution], J, window_size) #changed index of U here since we evaluate the smoothing
                if(print_flag):
                    print('j,q,res = ' + repr((j,q,resolution))+'--len filters=' + repr(filtersjq.shape) + '--filtered siglen = ' + repr(x_conv.shape) +'--phi_len = ' + repr(len(wavelet_filters['phi'][resolution])) + '--Max='+repr(max(S[indx,:])))
                indx = indx + 1
        S_tree[1] = S1
        
        if(display_flag):
            plt.figure()
            plt.imshow(np.log(filtmat+1e-6),aspect='auto',cmap='jet')
            plt.title('Filtered signal before subsampling')

            plt.figure()
            plt.imshow(S1, aspect='auto', cmap='jet')
            plt.title('First Order Coeffs')    
            plt.xlabel('Time window')
            plt.ylabel('octave J')
            plt.colorbar()

    if M > 1: #Second order scattering coeffs
        num_order2_coefs = int(J*(J-1)*Q**2/2)
        S2 = S[J*Q+1:num_coefs, :].view()  # view of the data
        S2.shape = (num_order2_coefs, window_size)
        indx = 0
        for j1 in range(J):
            #pick resolution of filtered signal stored during U1 calculation.
            current_resolution = v_resolution[j1] 
            for q1 in range(Q):
                #U is in the time domain                
                Ujq = fft_module.fft(U[j1*Q+q1])  
                for j2 in range(j1+1,J):
                    for q2 in range(Q):# TODO
                        # | U_lambda1 * Psi_j2 | * phi
                        filtersj2q2 = wavelet_filters['psi'][(j2,q2)][current_resolution].view()
                        #why no subsampling here ?? while we do it in 1st order 
                        x_conv = np.abs(fft_module.ifft(Ujq*filtersj2q2))               
                        Uj2 = _apply_lowpass(x_conv, wavelet_filters['phi'][current_resolution], J,  window_size)
                        if(print_flag):
                            print('j1,q1,j2,q2,res :' + repr((j1,q1,j2,q2,resolution)) + 'Uj2:'+repr(Uj2.shape) + 'x_conv=' + repr(x_conv.shape))
                        S2[indx, :] = Uj2
                        indx = indx+1
       # save tree structure
        S_tree[2] = S2

        if(print_flag):
            print('Number of (j1,j2) coeffs =' + repr(indx) + 'num_order2_coefs='+repr(num_order2_coefs))
        
        if(display_flag):
            plt.figure()
            plt.imshow(S2, aspect='auto', cmap='jet')
            plt.title('Second Order Coeffs')
            plt.colorbar()

    return S, U, S_tree

def get_audio_test():
    """ load test file from librosa/tests/data/test1_22050.wav """
    (sr, y) = scipy.io.wavfile.read('test1_22050.wav')
    y = y.mean(axis=1)
    return (y, sr)
    
def get_chirp(N):
    """Create a chirp signal of length N with 0 start freq and f1 at time t1 
    """
    t = np.linspace(0, 20, N)
    y = chirp(t, f0=0, f1=N/16, t1=N/16, method='linear')
    return y
    
def get_dirac(N, loc):
    """ Create dirac function at location loc of length N """
    y = np.zeros(N)
    y[loc] = 1
    return y
    
def test_scattering(bins_per_octave, quality_factor, nOctaves, N, M):
    """Test scattering transform 
    Parameters
    ----------
    bins_per_octave : integer
        Number of wavelet filters per octave in the fourier domain.
    quality_factor : integer
        The quality factor for all wavelet filters (by default 1)
    nOctaves : integer
        Number of Octaves or scales 
    N : integer
        length of input signal ( this is just for testing)
    M : integer
        scattering order (can be 1 or 2)
    
    """
    
    #Read / Create signals 
    
    # uncomment to test chirp and dirac signals
#    (y, fs) = get_audio_test()
    y = get_dirac(N, int(N/8))
#    y = get_chirp(N)
#    y, fs = librosa.load(librosa.util.example_audio_file())
       
    x = np.zeros(shape=(N))
    x = y[0:N]
    
    assert(nOctaves < np.log2(N))
    
    #scattering transform
    psi_specs = get_wavelet_filter_specs(bins_per_octave, quality_factor, nOctaves)
    filters, lp = filterbank_morlet_1d(N, psi_specs, nOctaves)
    wavelet_filters = filterbank_to_multiresolutionfilterbank(filters, nOctaves)
    scat,u,scat_tree = scattering(x, wavelet_filters=wavelet_filters, M=M)
    coef_index, spatial = scat.shape
    
    #CQTs and MFCCs
    cqts = librosa.cqt(x, sr=fs)
#    mfccs = librosa.feature.mfcc(y=y, sr=fs)
    
    
    if(display_flag):        
        #Librosa's Constant Q Transform 
        plt.figure()
        plt.imshow(cqts, aspect='auto',origin='lower')
        plt.colorbar()
        plt.title('Constant-Q power spectrum')
        plt.xlabel('Time windows')
        plt.ylabel('Log Freq.')
        
        plt.figure()
        plt.plot(x)
        plt.title('Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    
#        plt.figure()
#        plt.title('Scattering Transform')
#        plt.xlabel('time')
#        plt.ylabel('$\lambda$')
#        plt.imshow(np.log2(scat+1e-6), aspect='auto', cmap='jet')#add log to scattering coefficients view better
#        plt.colorbar()
    
    return scat, mfccs, cqts



bins_per_octave = 1
nOctaves = 12
N = 2**16
quality_factor = 1 #this parameter is never used (remove it)
M = 1

plt.close('all')

test_args = {"bins_per_octave": bins_per_octave, 
             "quality_factor": quality_factor, 
             "nOctaves":nOctaves, 
             "N":N, 
             "M":M
             }


global display_flag, debug_flag
display_flag = 1
debug_flag = 0
print_flag =  0

#psi_specs = get_wavelet_filter_specs(bins_per_octave, 1, nOctaves)
#filters, lp = filterbank_morlet_1d(N, psi_specs, nOctaves)
#wavelet_filters = filterbank_to_multiresolutionfilterbank(filters, nOctaves)
scat, mfccs, cqts = test_scattering(**test_args)

            