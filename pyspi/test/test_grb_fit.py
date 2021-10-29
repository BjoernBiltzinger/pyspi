
def test_grb_fit():
    from pyspi.SPILike import SPILikeGRB
    from threeML import DataList
    from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
    from pyspi.utils.response.spi_response import ResponseRMFGenerator
    from pyspi.utils.response.spi_drm import SPIDRM
    from astropy.time import Time
    from astromodels import (Powerlaw, Log_uniform_prior, Uniform_prior,
                             PointSource, Model)
    import numpy as np
    from pyspi.utils.function_utils import find_response_version
    from pyspi.utils.response.spi_response_data import ResponseDataRMF
    from threeML import BayesianAnalysis

    ein = np.geomspace(20, 800, 300)
    ebounds = np.geomspace(20, 400, 30)
    grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
    version = find_response_version(grbtime)
    rsp_base = ResponseDataRMF.from_version(version)
    active_time = "65-75"
    bkg_time1 = "-500--10"
    bkg_time2 = "150-1000"
    spilikes = []
    ra_val = 94.6783
    dec_val = -70.99905
    for d in [0, 12, 18]:
        drm_generator = ResponseRMFGenerator.from_time(grbtime,
                                                       d,
                                                       ebounds,
                                                       ein,
                                                       rsp_base)
        sd = SPIDRM(drm_generator, ra_val, dec_val)
        tsb = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{d}",
                                                d,
                                                grbtime,
                                                response=sd,
                                                sgl_type="both",
                                                )
        tsb.set_active_time_interval(active_time)
        tsb.set_background_interval(bkg_time1, bkg_time2)

        sl = tsb.to_spectrumlike()
        spilikes.append(SPILikeGRB.from_spectrumlike(sl,
                                                     free_position=False))
    datalist = DataList(*spilikes)

    pl = Powerlaw()
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e4)
    pl.K.bounds = (1e-6, 1e4)
    pl.index.set_uninformative_prior(Uniform_prior)
    pl.piv.value = 200
    ps = PointSource('GRB', ra=ra_val, dec=dec_val, spectral_shape=pl)

    model = Model(ps)

    ba_spi = BayesianAnalysis(model, datalist)
    ba_spi.set_sampler("emcee", share_spectrum=True)
    ba_spi.sampler.setup(n_walkers=20, n_iterations=500, seed=1000)
    ba_spi.sample()

    expected_result = [0.02192459821838934, -1.0371965175849485]

    assert np.isclose(
        ba_spi.results.get_data_frame()["value"]
        ["GRB.spectrum.main.Powerlaw.K"],
        expected_result[0],
        rtol=0.1,
    )

    assert np.isclose(
        ba_spi.results.get_data_frame()["value"]
        ["GRB.spectrum.main.Powerlaw.index"],
        expected_result[1],
        rtol=0.1,
    )
