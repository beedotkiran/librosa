def getMotherFrequency(bins_per_octave):
	"""
	The dimensionless mother center frequency '\xi' (corresponding to a log-period
	'\gamma=0') is computed as the midpoint between the center frequency of the second
	center frequency '\xi*2^(-1/bins_per_octave)' (corresponding to '\gamma=1') and the
	negative mother center frequency '(1-\xi)'. Hence the equation
	'2\xi = \xi*2^(-1/bins_per_octave) + (1-\xi)', of which we
	derive '\xi = 1 / (3 - 2^(1/bins_per_octave))'. This formula is valid
	only when the wavelet is a symmetric bump in the Fourier domain.
	
	Function returns the dimensionless mother frequency 
	
	Inputs 
	======
	bins_per_octave : Number of Filters per Octave 
	
	Outputs
	=======
	motherFrequency : dimensionaless mother center frequency
	
	"""

	motherFrequency = 1/(3-2**(-1/bins_per_octave))
	return motherFrequency