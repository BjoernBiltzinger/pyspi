try:
    import numpy as np
    has_numpy = True
except:
    has_numpy = False
try:
    from numba import njit, vectorize, cuda, prange, float64, int64
    has_numba = True
    has_cuda = cuda.is_available()
except:
    has_numba = False
    has_cuda = False
    
from math import log, pi, lgamma
from threeML.plugins.gammaln import logfactorial
#TODO AT the moment all - cash to use in multinest (multinest maximizes...)
if has_numpy:
    def cash_numpy(observed_counts, expected_model_counts):
        """
        Numpy implentation of Cash statistics. 
        :param observed_counts: Observed counts
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """
        return -np.sum(expected_model_counts-observed_counts*np.log(expected_model_counts))

if has_numba:

    @njit([float64(float64[:],  float64[:])])
    def cash_numba_cpu(observed_countsa, expected_model_countsa):
        """
        Numba cpu implentation of Cash statistics without parallel
        :param observed_counts: Observed counts
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        result = np.zeros(observed_countsa.size)

        for i in prange(len(observed_countsa)):
            
            observed_counts, expected_model_counts = observed_countsa[i], expected_model_countsa[i]

            result[i] = expected_model_counts-observed_counts*np.log(expected_model_counts)
            
        return -np.sum(result)    

    @njit([float64(float64[:], float64[:])], parallel=True)
    def cash_numba_cpu_par(observed_countsa, expected_model_countsa):
        """
        Numba implentation of Cash statistics with parallel
        :param observed_counts: Observed counts
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """
        result = np.zeros(observed_countsa.size)

        for i in prange(len(observed_countsa)):
            
            observed_counts, expected_model_counts = observed_countsa[i], expected_model_countsa[i]

            result[i] = expected_model_counts-observed_counts*np.log(expected_model_counts)
            
        return -np.sum(result)    

    if has_cuda:
        
        @cuda.jit
        def cash_cuda_kernel(observed_counts, background_counts, background_error, expected_model_counts, result):
            pass

        def cash_numba_cuda(observed_counts, background_counts, background_error, expected_model_counts):
            """
            Cuda implentation of Cash statistics
            :param observed_counts: Observed counts
            :param expected_model_counts: Expected model counts for the current model setup
            :return: sum of loglike values
            """

            #TODO write this

            return result.copy_to_host()[0]
