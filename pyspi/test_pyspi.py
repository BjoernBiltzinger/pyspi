from pyspi.SPILike_general import *
from pyspi.spi_analysis import *
from pyspi.Config_Builder import *
from threeML import *
conf = basic_config_builder_GRB()
conf.change_config(Simmulate=False,
                   Active_Time='117-124',
                   Background_time_interval_1='0-80',
                   Background_time_interval_2='150-320',
                   Time_of_GRB_UTC= '181201 023800',
                   Detectors_to_use='All',
                   Event_types=['single', 'double']) 



conf.display()
#spilike = SPILike('test',conf.config)
data = DataList(SPILike('test',conf.config))

pl = Powerlaw()
pl.piv =100
pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1000)
pl.index.set_uninformative_prior(Uniform_prior)

band = Band()                                                                                                                      
band.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1000)                                                               
band.xp.prior = Uniform_prior(lower_bound=10, upper_bound=8000)                                                                    
band.alpha.set_uninformative_prior(Uniform_prior)                                                                                  
band.beta.set_uninformative_prior(Uniform_prior)

a = PointSource('abc',ra=319.3,dec=-12.62,spectral_shape=band)


"""
a.position.ra.free = True
a.position.dec.free = True
a.position.ra.prior = Uniform_prior(lower_bound=0.0, upper_bound=360)
a.position.dec.prior = Cosine_Prior(lower_bound=-90.0, upper_bound=90)

model = Model(a)

ba = BayesianAnalysis(model, data)

wrap = [0] * len(model.free_parameters)
wrap[0]=1
"""

a.position.ra.free = False
a.position.dec.free = False


model = Model(a)

ba = BayesianAnalysis(model, data)

#wrap = [0] * len(model.free_parameters)
#wrap[0]=1


ba.sample_multinest(800, chain_name='./chain3', resume=False,
                    #wrapped_params=wrap,
                    verbose=True, importance_nested_sampling=False)
