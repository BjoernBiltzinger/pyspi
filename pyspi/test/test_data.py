# Test SpiData GRB
def test_spidata_grb():
    from pyspi.spi_data import SpiData_GRB
    from datetime import datetime
    from astropy.time.core import Time
    import numpy as np

    grb_time = '120711 024448'
    time = datetime.strptime(grb_time, '%y%m%d %H%M%S')
    grb_time = Time(time)

    data = SpiData_GRB(grb_time, afs=False, event_types=['single', 'double', 'triple'])

    ebins = np.logspace(np.log10(20), np.log10(8000), 100)
    
    data.time_and_energy_bin_sgl(ebounds = ebins, time_bin_step=1., start=-100, stop=100)

    det_0_counts =data.energy_and_time_bin_sgl_dict[0]

    assert np.sum(det_0_counts)==8977, 'Test spidata_grb failed'

    assert np.all(data.ebounds==ebins), 'Test spidata_grb failed'

    assert data._pointing_id==118900580010, 'Test spidata_grb failed'

    
