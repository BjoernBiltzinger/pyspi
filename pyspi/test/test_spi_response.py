import numpy as np


from pyspi.spi_response import SPIResponse
from pyspi.io.package_data import get_path_of_data_file


def test_response_constructor():

    response = SPIResponse()

    assert response.irfs is not None

def test_positional_components():

    response = SPIResponse()

    x, y = response.get_xy_pos(0.,0)





def test_roland():


    response = SPIResponse()

    response.rod