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

if has_numpy:
    def pgstat_numpy(observed_counts, background_counts, background_error, expected_model_counts):
        """
        Numpy implentation of PG-Stat. Taken from threeML/plugins/OGIP/likelihood_functions.py.
        :param observed_counts: Observed counts
        :param background_counts: Estimated background counts
        :param background_error: Error on background counts estimation
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        MB = background_counts + expected_model_counts
        s2 = background_error ** 2

        b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * observed_counts) + background_error ** 4)
                   + background_counts - expected_model_counts - s2)

        # Now there are two branches: when the background is 0 we are in the normal situation of a pure
        # Poisson likelihood, while when the background is not zero we use the profile likelihood

        # NOTE: bkgErr can be 0 only when also bkgCounts = 0
        # Also it is evident from the expression above that when bkgCounts = 0 and bkgErr=0 also b=0

        # Let's do the branch with background > 0 first

        idx = background_counts > 0

        log_likes = np.empty_like(expected_model_counts)

        log_likes[idx] = (-(b[idx] - background_counts[idx]) ** 2 / (2 * s2[idx])
                          + observed_counts[idx] * np.log(b[idx] + expected_model_counts[idx])
                          - b[idx] - expected_model_counts[idx] - logfactorial(observed_counts[idx])
                          - 0.5 * log(2 * np.pi) - np.log(background_error[idx]))

        # Let's do the other branch

        nidx = ~idx

        # the 1e-100 in the log is to avoid zero divisions
        # This is the Poisson likelihood with no background
        log_likes[nidx] = observed_counts[nidx] * np.log(expected_model_counts[nidx] + 1e-100) - \
                          expected_model_counts[nidx] - logfactorial(observed_counts[nidx])
        return np.sum(log_likes)

if has_numba:
    @vectorize('float64(float64)')
    def log_factorial(x):
        return lgamma(x+1)

    @njit([float64(float64[:], float64[:], float64[:], float64[:])])
    def pgstat_numba_cpu(observed_countsa, background_countsa, background_errora, expected_model_countsa):
        """
        Numba cpu implentation of PG-Stat without parallel
        :param observed_counts: Observed counts
        :param background_counts: Estimated background counts
        :param background_error: Error on background counts estimation
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        result = np.zeros(observed_countsa.size)

        for i in prange(len(observed_countsa)):

            observed_counts, background_counts, background_error, expected_model_counts = observed_countsa[i], background_countsa[i], background_errora[i], expected_model_countsa[i]

            if background_counts>0:

                MB = background_counts + expected_model_counts
                s2 = background_error ** 2

                b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * observed_counts) +
                                   background_error ** 4) + background_counts -
                           expected_model_counts - s2)


                result[i] = (-(b - background_counts) ** 2 / (2 * s2)
                              + observed_counts * np.log(b + expected_model_counts)
                              - b - expected_model_counts - log_factorial(observed_counts)
                              - 0.5 * log(2 * np.pi) - np.log(background_error))
            else:
                result[i] = observed_counts * np.log(expected_model_counts + 1e-100) - \
                          expected_model_counts - log_factorial(observed_counts)
        return np.sum(result)    

    @njit([float64(float64[:], float64[:], float64[:], float64[:])], parallel=True)
    def pgstat_numba_cpu_par(observed_countsa, background_countsa, background_errora, expected_model_countsa):
        """
        Numba implentation of PG-Stat with parallel
        :param observed_counts: Observed counts
        :param background_counts: Estimated background counts
        :param background_error: Error on background counts estimation
        :param expected_model_counts: Expected model counts for the current model setup
        :return: sum of loglike values
        """

        result = np.zeros(observed_countsa.size)

        for i in prange(len(observed_countsa)):

            observed_counts, background_counts, background_error, expected_model_counts = observed_countsa[i], background_countsa[i], background_errora[i], expected_model_countsa[i]

            if background_counts>0:

                MB = background_counts + expected_model_counts
                s2 = background_error ** 2

                b = 0.5 * (np.sqrt(MB ** 2 - 2 * s2 * (MB - 2 * observed_counts) +
                                   background_error ** 4) + background_counts -
                           expected_model_counts - s2)


                result[i] = (-(b - background_counts) ** 2 / (2 * s2)
                              + observed_counts * np.log(b + expected_model_counts)
                              - b - expected_model_counts - log_factorial(observed_counts)
                              - 0.5 * log(2 * np.pi) - np.log(background_error))
            else:

                result[i] = observed_counts * np.log(expected_model_counts + 1e-100) - \
                          expected_model_counts - log_factorial(observed_counts)

        return np.sum(result)    

    if has_cuda:
        
        @cuda.jit
        def pgstat_cuda_kernel(observed_counts, background_counts, background_error, expected_model_counts, result):
            pass

        def pgstat_nuba_cuda(observed_counts, background_counts, background_error, expected_model_counts):
            """
            Cuda implentation of PG-Stat
            :param observed_counts: Observed counts
            :param background_counts: Estimated background counts
            :param background_error: Error on background counts estimation
            :param expected_model_counts: Expected model counts for the current model setup
            :return: sum of loglike values
            """
            observed_counts_d = cuda.to_device(observed_counts)
            background_counts_d = cuda.to_device(background_counts)
            background_error_d = cuda.to_device(background_error)
            expected_model_counts_d = cuda.to_device(expected_model_counts)

            threadsperblock = 512
            blockspergrid = (observed_counts.size + (threadsperblock - 1)) // threadsperblock
            result = cuda.device_array([1,])
            init = [0.]
            result = cuda.to_device(init)
            pgstat_cuda_kernel[blockspergrid, threadsperblock](observed_counts_d, background_counts_d, background_error_d, expected_model_counts_d, result)

            return result.copy_to_host()[0]
