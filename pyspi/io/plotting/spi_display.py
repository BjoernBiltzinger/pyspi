import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import RegularPolygon
from astromodels import *
from pyspi.utils.livedets import get_live_dets
from pyspi.io.array_to_cmap import array_to_cmap

NUM_REAL_DETS = 19
NUM_PSEUDO_DOUBLE_DETS = 42
NUM_PSEUDO_TRIPLE_DETS = 42
NUM_TOTAL_DETS = NUM_REAL_DETS + NUM_PSEUDO_DOUBLE_DETS + \
    NUM_PSEUDO_TRIPLE_DETS


# the origins of the detectors
# the underscore keeps these variable from being exposed
# to the user
_detector_origins = ((0, 0), (6, 0), (3, 5.196),
                     (-3, 5.196), (-6, 0), (-3, -5.196),
                     (3, -5.196), (9, -5.196), (12, 0),
                     (9, 5.196), (6, 10.392), (0, 10.392),
                     (-6, 10.392), (-9, 5.196), (-12, 0),
                     (-9, -5.196), (-6, -10.392), (0, -10.392),
                     (6, -10.392))

def _calc_double_origin(det1, det2):
    x = (_detector_origins[det1][0] + _detector_origins[det2][0]) * 0.5
    y = (_detector_origins[det1][1] + _detector_origins[det2][1]) * 0.5

    return x, y

def _construct_double_events_table():
    """

    Helper function to generate double event detector list


    :return:
    """

    # the list of double pairs
    # for details see:
    # https://heasarc.gsfc.nasa.gov/docs/integral/spi/pages/detectors.html
    double_event_pairs = (
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 6),
        (1, 7), (1, 8), (1, 9), (2, 3), (2, 9), (2, 10), (2, 11), (3, 4),
        (3, 11), (3, 12), (3, 13), (4, 5), (4, 13), (4, 14), (4, 15), (5, 6),
        (5, 15), (5, 16), (5, 17), (6, 7), (6, 17), (6, 18), (7, 8), (7, 18),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        (14, 15), (15, 16), (16, 17), (17, 18))

    # calculate the origins

    origins = np.array([_calc_double_origin(*pair)
                        for pair in double_event_pairs])

    # build a dictionary for out put

    out = dict(detector_number=range(19, 19 + 42),
               x=origins[:, 0],
               y=origins[:, 1],
               detector1=np.array(double_event_pairs)[:, 0],
               detector2=np.array(double_event_pairs)[:, 1]

               )

    # return the as a pandas data frame

    return pd.DataFrame(out)


# build a list that can be exposed to the user
spi_pseudo_double_detectors = _construct_double_events_table()


class SPIDetector(object):

    def __init__(self, detector_number, origin, is_pseudo_detector=False):
        """
        A SPI detector is defined by its number, origin and type
        :param detector_number: the detector number
        :param origin: the detector origin
        :param is_pseudo_detector: if this is a real detector or not
        """
        self._detector_number = detector_number
        self._origin = origin
        self._is_pseudo_detector = is_pseudo_detector
        self._bad = False
        
    @property
    def bad(self):
        return self._bad

    def set_bad(self, flag):
        """
        Set the flag if this is a bad detector
        :param flag: Bad detector?
        :return: 
        """
        self._bad = flag
    
    @property
    def origin(self):
        return self._origin

    @property
    def is_pseudo_detector(self):
        return self._is_pseudo_detector

    @property
    def detector_number(self):
        return self._detector_number


class DoubleEventDetector(SPIDetector):

    def __init__(self, detector_number, origin, detector1, detector2):
        """

        :param detector_number:
        :param origin:
        :param detector1:
        :param detector2:
        """

        super(DoubleEventDetector, self).__init__(detector_number, origin, is_pseudo_detector=True)

        self._detector1 = detector1
        self._detector2 = detector2


# Fill out this class

class TripleEventDetector(SPIDetector):
     pass



class DetectorContents(object):

    def __init__(self, detector_array):
        assert len(detector_array) == NUM_TOTAL_DETS

        self._contents = np.array(detector_array)

        self._real_contents = np.array(detector_array[:NUM_REAL_DETS])

    @classmethod
    def from_spi_data(cls, spi_data):
        pass

    @classmethod
    def from_total_effective_area(cls, spi_response, azimuth, zenith):
        effective_area = spi_response.effective_area_per_detector(azimuth, zenith).sum(axix=0)

        return cls(effective_area)


class SPI(object):

    def __init__(self, bad_detectors=[], time=None):

        self._bad_detectors = bad_detectors
        if time:
            good_detectors = get_live_dets(time)
            self._bad_detectors = [i
                                   for i in range(19)
                                   if i not in good_detectors]
        self._construct_detectors()

    def _construct_detectors(self):
        """

        :return:
        """

        # the real detector origins
        self._detector_origins = _detector_origins
        # go through an build the list of detectors for SPI
        self._detectors = []

        n = 0  # keeps track of the detector number

        for origin in self._detector_origins:

            # first we build the real detectors

            self._detectors.append(SPIDetector(detector_number=n,
                                               origin=origin,
                                               is_pseudo_detector=False))

            n += 1

        # now we build the double event detectors

        for detector in spi_pseudo_double_detectors.iterrows():

            detector = detector[1]

            origin = (detector['x'], detector['y'])

            self._detectors.append(DoubleEventDetector(detector_number=n,
                                                       origin=origin,
                                                       detector1=detector['detector1'],
                                                       detector2=detector['detector2']))

            n += 1

        # TODO: Add the triple event detectors


        for det in self._detectors:
            if det.detector_number in self._bad_detectors:
                det.set_bad(True)

    def plot_spi_working_dets(self,
                              with_pseudo_detectors=True,
                              show_detector_number=True):
        """
        Plot the SPI Detectors and mark the detectors that are not working red
        :param with_pseudo_detectors: Plot pseudo detectors?
        :param show_detector_number: Show the det numbers in the plot?
        :return:
        """
        

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=(15,15))

        d=1

        radius = d*2* 1.732

        # first get the colors based of the contents
        #colors, pseudo_colors = self._get_colors_from_contents(cmap,
        #                                                       pseudo_cmap)
        # color iterators
        n = 0
        pseudo_n = 0

        # now we loop over all the detectors and if they have contents
        # we will plot them
        for detector in self._detectors:
            # first the real detectors

            if not detector.is_pseudo_detector:

                if detector.bad:
                    color = "red"
                else:
                    color = "green"

                # create a ploygon and color it based of the contents
                p = RegularPolygon(xy=tuple(d*i for i in detector.origin),
                                   numVertices=6,
                                   radius=radius,
                                   facecolor=color,
                                   ec='k',
                                   lw=3)

                ax.add_patch(p)

                # show the detector number
                if show_detector_number:
                    ax.text(d*detector.origin[0],
                            d*detector.origin[1],
                            detector.detector_number,
                            ha="center",
                            va="center",
                            color='k',
                            size=30)

        ax.set_xlim(d*-16, d*16)
        ax.set_ylim(d*-16, d*16)

        ax.set_yticks([])
        ax.set_xticks([])
        return fig

    def _loadtxt2d(self, intext):
        try:
            return np.loadtxt(intext, ndmin=2)
        except:
            return np.loadtxt(intext)

