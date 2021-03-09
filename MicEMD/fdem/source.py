__all__ = ['Source']
import SimPEG.electromagnetics.frequency_domain as fdem
from abc import ABCMeta
from abc import abstractmethod


class BaseSource(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, Target, Detector, Collection, *args):
        self.target = Target
        self.detector = Detector
        self.collection = Collection

    @abstractmethod
    def source_list(self):
        pass


class Source(BaseSource):

    def __init__(self, Target, Detector, Collection, *args):
        BaseSource.__init__(self, Target, Detector, Collection)

    @property
    def source_list(self):
        # Create empty list to store sources
        frequencies = [self.detector.frequency]
        receiver_locations = self.collection.receiver_location
        source_locations = self.collection.source_locations
        _source_list = []
        ntx = source_locations.shape[0]

        # Each unique location and frequency defines a new transmitter
        for ii in range(len(frequencies)):
            for jj in range(ntx):
                # Define receivers of different type at each location
                bxr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "x", "real"
                )
                bxi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "x", "imag"
                )
                byr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "y", "real"
                )
                byi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "y", "imag"
                )
                bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "z", "real"
                )
                bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                    receiver_locations[jj, :], "z", "imag"
                )
                receivers_list = [bxr_receiver, bxi_receiver, byr_receiver,
                                  byi_receiver, bzr_receiver, bzi_receiver]

                # Must define the transmitter properties and associated receivers
                _source_list.append(
                    fdem.sources.MagDipole(
                        receivers_list,
                        frequencies[ii],
                        source_locations[jj],
                        orientation="z",
                        moment=self.detector.mag_moment
                    )
                )
        return _source_list





