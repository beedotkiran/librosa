def getMotherFrequency(NFO):
	"""
	The dimensionless mother center frequency `ξ` (corresponding to a log-period
	`γ=0`) is computed as the midpoint between the center frequency of the second
	center frequency `ξ*2^(-1/nFilters_per_octave)` (corresponding to `γ=1`) and the
	negative mother center frequency `(1-ξ)`. Hence the equation
	`2ξ = ξ*2^(-1/nFilters_per_octave) + (1-ξ)`, of which we
	derive `ξ = 1 / (3 - 2^(1/nFilters_per_octave))`. This formula is valid
	only when the wavelet is a symmetric bump in the Fourier domain.
	
	Function returns the dimensionless mother frequency 
	
	Inputs 
	======
	NFO : Number of Filters per Octave (this is Q in the paper)
	Outputs
	=======
	motherFrequency
	
	"""

	motherFrequency = 1/(3-2**(-1/NFO))
	return motherFrequency



