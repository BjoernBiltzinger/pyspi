import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

from pyspi.io.array_to_cmap import array_to_cmap


NUM_REAL_DETS = 19
NUM_PSEUDO_DETS = 42
NUM_TOTAL_DETS = NUM_REAL_DETS + NUM_PSEUDO_DETS

class SPIDetector(object):

    def __init__(self, detector_number, origin, is_pseudo_detector=False):
        """
        A SPI detector is defined by its number, origin and type
        :param detector_number: the detector number
        :param origin: the detector origin
        :param is_pseudo_detector: if this is a real detector or not
        """
        self._contents = None
        self._detector_number = detector_number
        self._origin = origin
        self._is_pseudo_detector = is_pseudo_detector

    @property
    def contents(self):
        return self._contents

    def set_contents(self, contents):
        """
        Fill the contents of the detector

        :param contents: a numeric value
        :return: None
        """


        self._contents = contents

    @property
    def origin(self):
        return self._origin

    @property
    def is_pseudo_detector(self):
        return self._is_pseudo_detector

    @property
    def detector_number(self):
        return self._detector_number


class DetectorContents(object):

    def __init__(self, detector_array):

        assert len(detector_array) == NUM_TOTAL_DETS

    @classmethod
    def from_spi_data(cls, spi_data):
        pass

class SPI(object):

    def __init__(self):


        self._bad_detectors = [2, 5]

        self._construct_detectors()

    def _construct_detectors(self):
        """

        :return:
        """

        # the real detector origins
        self._detector_origins = ((0, 0), (6, 0), (3, 5.196),
                                  (-3, 5.196), (-6,0), (-3, -5.196),
                                  (3, -5.196), (9, -5.196), (12, 0),
                                  (9, 5.196), (6, 10.392), (0, 10.392),
                                  (-6, 10.392), (-9, 5.196), (-12, 0),
                                  (-9, -5.196), (-6, -10.392),  (0, -10.392),
                                  (6, -10.392))

        # the pseudo detector origins
        self._pseudo_detector_origins = ((3, 0),
                                         (1.5, 2.598), (-1.5, 2.598), (-3, 0),
                                         (-1.5, -2.598), (1.5, -2.598),
                                         (4.5, 2.598), (4.5, -2.598), (7.5, -2.598),
                                         (9, 0), (7.5, 2.598),
                                         (0, 5.196), (6, 5.196), (4.5, 7.794),
                                         (1.5, 7.794),
                                         (-4.5, 2.598),
                                         (-1.5,  7.794), (-4.5, 7.794), (-6, 5.196),
                                         (-4.5, -2.598), (-7.5,  2.598), (-9, 0),
                                         (-7.5, -2.598),
                                         (0, -5.196), (-6, -5.196), (-4.5, - 7.794),
                                         (-1.5, -7.794),
                                         (6, -5.196), (1.5, -7.794), (4.5, -7.794),
                                         (10.5, -2.598), (7.5, -7.794),
                                         (10.5, 2.598), (7.5, 7.794), (3, 10.392),
                                         (-3, 10.392), (-7.5, 7.794),
                                         (-10.5, 2.598), (-10.5, -2.598), (-7.5, -7.794),
                                         (-3, -10.392), (3, -10.392))

        # go through an build the list of detectors for SPI
        self._detectors = []

        n = 0 # keeps track of the detector number

        for origin in self._detector_origins:
            self._detectors.append( SPIDetector(detector_number=n, origin=origin, is_pseudo_detector=False) )

            n += 1

        for origin in self._pseudo_detector_origins:
            self._detectors.append(SPIDetector(detector_number=n, origin=origin, is_pseudo_detector=True))
            n += 1

        # REMOVE!!!!!!!
        ii = 0
        for det in self._detectors:
            if det.detector_number not in self._bad_detectors:
                det.set_contents(ii)
            ii += 1

    def _get_colors_from_contents(self, cmap, pseudo_cmap):
        """

        :param cmap: colormap to use for real detectors
        :param pseudo_cmap: colormap to use for pseudo detectors
        :return: tuple of color arrays
        """

        contents = []
        pseudo_contents = []

        for detector in self._detectors:

            if detector.contents is not None:

                if detector.is_pseudo_detector:

                    pseudo_contents.append(detector.contents)

                else:

                    contents.append(detector.contents)

        _, colors = array_to_cmap(np.array(contents), cmap=cmap)
        _, pseudo_colors = array_to_cmap(np.array(pseudo_contents), cmap=pseudo_cmap)

        return colors, pseudo_colors

        _, colors = array_to_cmap()

    def fill_detectors(detector_contents):
        """

        :return:
        """

        pass

    def plot_spi(self, with_pseudo_detectors=True,
                 show_detector_number=True,
                 cmap='viridis',
                 pseudo_cmap='plasma'):

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

        # the radius of the polygons must change if we only
        # show real detectors
        if with_pseudo_detectors:

            radius = 1.732

        else:

            radius = 2 * 1.732

        # first get the colors based of the contents
        colors, pseudo_colors = self._get_colors_from_contents(cmap,
                                                               pseudo_cmap)
        # color iterators
        n = 0
        pseudo_n = 0

        # now we loop over all the detectors and if they have contents
        # we will plot them
        for detector in self._detectors:

            # first the real detectors

            if not detector.is_pseudo_detector:

                if detector.contents is not None:

                    # create a ploygon and color it based of the contents

                    p = RegularPolygon(xy=detector.origin, numVertices=6, radius=radius,
                                       facecolor=colors[n], ec='k', lw=3)

                    ax.add_patch(p)

                    # show the detector number
                    if show_detector_number:
                        ax.text(detector.origin[0], detector.origin[1], detector.detector_number,
                                ha="center", va="center", color='k', size=14)

                    n += 1

            # now the pseudo detectors if we have chosen to plot them

            if detector.is_pseudo_detector and with_pseudo_detectors:

                if detector.contents is not None:
                    p = RegularPolygon(xy=detector.origin,
                                       numVertices=6,
                                       radius=radius,
                                       facecolor=pseudo_colors[pseudo_n], alpha=.5)

                    ax.add_patch(p)

                    # show the detector number
                    if show_detector_number:
                        ax.text(detector.origin[0], detector.origin[1], detector.detector_number, ha="center", va="center",
                                color='k', size=14)

                    pseudo_n += 1


        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)

        ax.set_yticks([])
        ax.set_xticks([])
