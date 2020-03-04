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
    
from pyspi.utils.likelihood_functions.pgstat import *
from pyspi.utils.likelihood_functions.cash import *

class Likelihood(object):

    def __init__(self, n_likelihoods=None, numba_cuda=None, numba_cpu=None, numpy=None, parallel=True):
        """
        Init likelihood object. This can be used for PGStat and Cstat. Supports numba and cuda
        if available.
        :param n_likelihoods: Number of single likelihoods to calculate for every total likelihood evaluation
        :param numba_cuda: Flag to agressively use or not use cuda. If None the object will figure out by its own if cuda is usefull here
        :param numba_cpu: Flag to agressively use or not use numba_cpu. If None the object will figure out by its own if numba cpu is usefull here
        """

        assert numba_cpu != True or numba_cuda!= True, 'You can only set one of the numba_cuda/numba_cpu flag to True. I can not use both.'

        self._parallel = parallel
                    
        # Figure out which approach should be used (numpy, numba_cpu or numba_cuda)

        if (not numba_cuda) and (not numba_cpu) and (not numpy):
            
            if n_likelihoods>100000 and has_cuda:
                print('Many single likelihood values have to be evaluated at once and CUDA is available. I will use this to speed up the likelihood evaluations!')
                numba_cuda = True
            else:
                if n_likelihoods==None:
                    print('Number of single likelihood evaluation not given. Can not figure out best way to calculate Likelihood. I will simply use numba_cpu if available otherwise numpy.')
                elif n_likelihoods<100000:
                    print('Not enough single likelihood evaluations to make CUDA usefull. I will simply use numba_cpu if available otherwise numpy.')
                else:
                    print('CUDA not available. I will simply use numba_cpu if available otherwise numpy.')
                if has_numba:
                    numba_cpu = True
                elif has_numpy:
                    numpy = True
                else:
                    raise NotImplementedError("Neither numpy nor numba are installed. Please install at least one of them.")
                
                    
                    
                
        self._numpy = False
        self._numba_cpu = False
        self._numba_cuda = False
                    
        if numba_cuda or numba_cpu:
            assert has_numba, 'Numba is not available. Please install it, or do not set numba_cuda/numba_cpu to True. Then numpy will be used.'
            if numba_cuda:
                assert has_cuda, 'Cuda is not available on this system. Please set it up or use numba_cpu or no flag at all.'
                self._numba_cuda = True
            else:
                self._numba_cpu = True
        elif numpy:
            assert has_numpy, 'Numpy is not available.'
            self._numpy = True 

        if self._numpy:
            print('Figured out which approach to use in the likelihood evaluation. I will use the numpy approach!')
            if parallel:
                print('No parallel support for numpy implemented.')

        if self._numba_cuda:
            print('Figured out which approach to use in the likelihood evaluation. I will use the numba_cuda approach!')

        if self._numba_cpu:
            print('Figured out which approach to use in the likelihood evaluation. I will use the numba_cpu approach!')

    def PG_stat(self, observed_counts, background_counts, background_error, expected_model_counts):
        """
        Poisson uncertainties on the observed coutns and Gaussian errors on background.
        Calls the defined implentations for numpy, numba_cpu or numba_cuda.
        :param observed_counts: Observed counts
        :param background_counts: Estimated background counts
        :param background_error: Error on background counts estimation
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        if self._numpy:
            return pgstat_numpy(observed_counts, background_counts, background_error, expected_model_counts)

        if self._numba_cpu:
            if self._parallel:
                return pgstat_numba_cpu_par(observed_counts, background_counts, background_error, expected_model_counts)
            else:
                return pgstat_numba_cpu(observed_counts, background_counts, background_error,expected_model_counts)

        if self._numba_cuda:
            return pgstat_numba_cuda(observed_counts, background_counts, background_error, expected_model_counts)

    
    def Cash_stat(self, observed_counts, expected_model_counts):
        """
        Cash statistics, no background and signal poisson distributed.
        Calls the defined implentations for numpy, numba_cpu or numba_cuda.
        :param observed_counts: Observed counts
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        if self._numpy:
            return cash_numpy(observed_counts, expected_model_counts)

        if self._numba_cpu:
            if self._parallel:
                return cash_numba_cpu_par(observed_counts, expected_model_counts)
            else:
                return cash_numba_cpu(observed_counts, expected_model_counts)

        if self._numba_cuda:
            return cash_numba_cuda(observed_counts, expected_model_counts)
