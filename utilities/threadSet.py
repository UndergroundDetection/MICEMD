from PyQt5.QtCore import QTranslator, QThread, pyqtSignal
import MicEMD.fdem as f
import numpy as np


class ThreadCalFdem(QThread):
    """Fdem forward simulation thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation.The simulation generates the
    forward data by the input condition of the simulation.

    Parameters
    ----------
    detector: class
        The detector class consists of the information about the detector
    target: class
        The target class consists of the information about the detector
    collection: class
        The collection class consists of the information about collecting
        the response of the target
    save: bool
        Whether to save the data of the forward simulation

    """

    # use the signal to get the result of the simulation thread
    trigger = pyqtSignal(tuple)

    def __init__(self):
        super(ThreadCalFdem, self).__init__()
        self.detector = None
        self.target = None
        self.collection = None
        self.save = None

    def run(self):
        """Use the information about the initial parameters to simulate"""

        forward_result = f.simulate(self.target, self.detector, self.collection, 'simpeg')
        self.trigger.emit(forward_result)
    # @property
    # def forward_result(self):
    #     return forward_result


class ThreadInvFdem(QThread):
    """Fdem inversion thread.
    Use the mutilThread to avoid that the mainWindow play died because of
    the long time of complicated calculation of the inversion.The inversion
    inverses by the input condition.

    Parameters
    ----------
    method: str
        The method of inversion
    iterations: int
        The maximum number of numeric optimization method
    tol: int
        The minimum cost of numeric optimization method
    x0: ndarray
        The initial parameters of inversion
    forward_result: tuple
        The result of simulation,consists of the information of the simulation
    save: bool
        Whether to save the result of inversion

    """
    # use the signal to get the result of the inversion thread
    trigger = pyqtSignal(dict)

    def __init__(self):
        super(ThreadInvFdem, self).__init__()
        self.method = None
        self.iterations = None
        self.tol = None
        self.x0 = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.forward_result = None
        self.target = None
        self.detector = None
        self.save = False

    def run(self):
        """
        Use the information about the condition of inversion to inverse,
        and utilize the signal to transfer the inv_result.

        """

        data = (self.forward_result[0], self.target, self.detector)
        inv_para = {'x0': self.x0, 'iterations': self.iterations, 'tol': self.tol}

        inv_result = f.inverse(data, self.method, inv_para)

        self.trigger.emit(inv_result)


